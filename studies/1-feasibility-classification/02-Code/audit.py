"""Independent raw-record and NumPy checkpoint audit; no training or model selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
info = dict(x.split(":", 1) for x in Path("/proc/meminfo").read_text().splitlines())
if int(info["MemAvailable"].split()[0]) < 2 * 1024**2:
    raise RuntimeError("Insufficient memory for checkpoint audit")
import numpy as np
import torch
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text())


def digest(state):
    h = hashlib.sha256()
    for name, values in state.items():
        h.update(name.encode())
        h.update(values.numpy().tobytes())
    return h.hexdigest()


def label(task, environment, query, a, b):
    fv, gv = {"E00": (0, 0), "update_f": (2, 0), "update_g": (0, 2)}[environment]
    f, g = task["f_tables"][fv], task["g_tables"][gv]
    if query == "f":
        return f[a]
    if query == "g":
        return g[b]
    return (f[a] + g[b]) % 8


def forward(state, rows):
    vectors = []
    for row in rows:
        x = [0.] * 19
        x[["f", "g", "combine"].index(row["query"])]=1.
        if row["a"] is not None:
            x[3+row["a"]]=1.
        if row["b"] is not None:
            x[11+row["b"]]=1.
        vectors.append(x)
    x = np.array(vectors, dtype=np.float64)
    for layer in (0, 2, 4):
        x = x @ state[f"{layer}.weight"].numpy().astype(np.float64).T + state[f"{layer}.bias"].numpy()
        if layer != 4:
            x = np.maximum(x, 0)
    return x


def category(row):
    if row["query"] != "combine":
        return row["query"]
    return "combined_heldout" if (row["b"]-row["a"]) % 8 in (0, 1) else "combined_seen"


def independent_metrics(rows, task, environment):
    result = defaultdict(lambda: [0, 0])
    for row in rows:
        target = label(task, environment, row["query"], row["a"], row["b"])
        group = result[category(row)]
        group[0] += int(row["predicted_class"] == target)
        group[1] += 1
    return {k:dict(correct=v[0], total=v[1], accuracy=v[0]/v[1]) for k, v in result.items()}


def load_records(path, task, ident, hashes):
    records=[]
    with gzip.open(path,"rt") as f:
        for line in f:
            row=json.loads(line)
            assert all(row[k]==v for k,v in ident.items()), "Wrong job identity"
            assert row["input_hashes"]==hashes, "Wrong source provenance"
            assert row["status"]=="ok"
            assert row["target_class"]==label(task,row["environment"],row["query"],row["a"],row["b"]), "Wrong symbolic label"
            assert row["split"] == ("heldout_pairs" if category(row)=="combined_heldout" else "train")
            logits=np.array(row["logits"])
            assert logits.shape==(8,) and np.isfinite(logits).all()
            assert row["predicted_class"]==int(logits.argmax()), "Prediction differs from logits"
            records.append(row)
    return records


def audit(run, analysis, output, repeat=None):
    run, analysis, output=Path(run).resolve(),Path(analysis).resolve(),Path(output).resolve()
    manifest=read(run/"manifest.json")
    root=run.parents[1]
    config=manifest["configuration"]
    for path,expected in manifest["input_hashes"].items():
        assert hashlib.sha256((run/"snapshot"/path).read_bytes()).hexdigest()==expected, "Snapshot hash changed"
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected, "Execution input changed"
    tasks={t["task_seed"]:t for t in read(run/"snapshot/03-Data/prepared/task-families.json")}
    summary=read(analysis/"summary.json")
    assert summary["source_sha256"]==hashlib.sha256((root/"02-Code/analyze.py").read_bytes()).hexdigest()
    paired_initials={}
    total_rows,total_checkpoints,total_trajectories,repeated_jobs=0,0,0,0
    max_forward_error=0.
    allbase=[]
    for job in manifest["jobs"]:
        directory=run/job["id"]
        assert read(directory/"status.json")["status"]=="complete"
        task=tasks[job["task_seed"]]
        ident={k:job[k] for k in ("task_seed","model_seed","history")}
        initial=torch.load(directory/"initial.pt",map_location="cpu",weights_only=True)
        key=(job["task_seed"],job["model_seed"])
        h=digest(initial["model"])
        assert paired_initials.setdefault(key,h)==h, "Histories did not start identically"
        base=torch.load(directory/"base.pt",map_location="cpu",weights_only=True)
        train=read(directory/"training.json")
        assert digest(base["model"])==train["final_hash"]==base["final_hash"]
        expected_steps=len(job["environment_order"])*config["steps_per_environment"]+config["final_extra_steps"]
        assert train["base_optimizer_steps"]==expected_steps
        assert all(int(s["step"])==expected_steps for s in base["optimizer"]["state"].values())
        base_rows=load_records(directory/"base-observations.jsonl.gz",task,ident,manifest["input_hashes"])
        assert len(base_rows)==80 and len({r["example_id"] for r in base_rows})==80
        counts=independent_metrics(base_rows,task,"E00")
        assert counts==train["blocks"][-1]["metrics"]
        allbase.append(dict(ident,metrics=counts))
        original_logits=np.array([r["logits"] for r in base_rows])
        reconstructed=forward(base["model"],base_rows)
        error=float(np.max(np.abs(reconstructed-original_logits)))
        max_forward_error=max(error,max_forward_error)
        np.testing.assert_allclose(reconstructed,original_logits,atol=2e-4,rtol=2e-5)
        np.testing.assert_array_equal(reconstructed.argmax(-1),original_logits.argmax(-1))
        total_checkpoints+=1
        raw=load_records(directory/"update-observations.jsonl.gz",task,ident,manifest["input_hashes"])
        groups=defaultdict(list)
        for row in raw:
            groups[(row["branch"],row["condition"],row["optimizer_mode"],row["step"])].append(row)
        updates=read(directory/"updates.json")["trajectories"]
        assert len(updates)==14
        assert len(raw)==14*7*80
        for tr in updates:
            primitive="f" if tr["branch"]=="update_f" else "g"
            env=tr["branch"] if tr["condition"]=="changed_primitive" else "E00"
            candidates=task["environments"][env]["examples"]
            if tr["condition"]=="no_update":
                expected_ids=[]
            elif tr["condition"]=="old_environment":
                expected_ids=[r["id"] for r in candidates if r["split"]=="train"]
            else:
                expected_ids=[r["id"] for r in candidates if r["query"]==primitive]
            for point in tr["curve"]:
                rows=groups[(tr["branch"],tr["condition"],tr["optimizer_mode"],point["step"])]
                assert len(rows)==80 and len({r["example_id"] for r in rows})==80
                assert all(r["training_example_ids"]==expected_ids for r in rows), "Wrong update examples"
                assert independent_metrics(rows,task,tr["branch"])==point["updated_target"]
                assert independent_metrics(rows,task,"E00")==point["old_target"]
                if point["step"]==0 or tr["condition"]=="no_update":
                    np.testing.assert_array_equal([r["logits"] for r in rows],original_logits)
            endpoint=groups[(tr["branch"],tr["condition"],tr["optimizer_mode"],100)]
            cp=torch.load(directory/tr["checkpoint"],map_location="cpu",weights_only=True)
            assert digest(cp["model"])==tr["final_hash"]==cp["final_hash"]
            assert tr["start_hash"]==base["final_hash"]
            expected=forward(cp["model"],endpoint)
            observed=np.array([r["logits"] for r in endpoint])
            max_forward_error=max(max_forward_error,float(np.max(np.abs(expected-observed))))
            np.testing.assert_allclose(expected,observed,atol=2e-4,rtol=2e-5)
            np.testing.assert_array_equal(expected.argmax(-1),observed.argmax(-1))
            delta=sum(float((cp["model"][k]-base["model"][k]).square().sum()) for k in base["model"])**.5
            assert abs(delta-tr["curve"][-1]["parameter_delta_l2"])<1e-5
            if tr["optimizer_mode"]=="none":
                assert cp["optimizer"] is None and digest(cp["model"])==base["final_hash"]
            else:
                steps=100+(expected_steps if tr["optimizer_mode"]=="retained" else 0)
                assert all(int(s["step"])==steps for s in cp["optimizer"]["state"].values())
            total_checkpoints+=1
            total_trajectories+=1
        total_rows+=len(raw)+len(base_rows)
        if repeat and (Path(repeat)/job["id"]).is_dir():
            prior=Path(repeat)/job["id"]
            assert read(prior/"training.json")["final_hash"]==train["final_hash"]
            old=read(prior/"updates.json")["trajectories"]
            assert [r["final_hash"] for r in old]==[r["final_hash"] for r in updates]
            for old_tr,new_tr in zip(old,updates):
                assert [p["updated_target"] for p in old_tr["curve"]]==[p["updated_target"] for p in new_tr["curve"]]
            repeated_jobs+=1
    for history,means in summary["base_history_means"].items():
        group=[r["metrics"] for r in allbase if r["history"]==history]
        for name,value in means.items():
            assert abs(sum(x[name]["accuracy"] for x in group)/len(group)-value)<1e-12
    result=dict(status="passed",run=run.name,model_jobs=len(allbase),raw_records_checked=total_rows,
                endpoint_checkpoints_checked=total_checkpoints,update_trajectories_checked=total_trajectories,
                maximum_numpy_forward_absolute_error=max_forward_error,qualification_jobs_exactly_reproduced=repeated_jobs,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--analysis",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--repeat",type=Path)
    a=p.parse_args()
    audit(a.run,a.analysis,a.output,a.repeat)
