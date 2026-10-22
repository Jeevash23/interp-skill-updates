# Reproduction-only derivative: changes output destination, not computation.
"""Post-confirmation fixed-readout duration control, independently computed in NumPy."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import audit as independent
import study as s
import amended_study as amended

ROOT=Path(__file__).resolve().parents[1]


def run():
    output=REPLAY_OUTPUT;output.mkdir(exist_ok=False)
    manifest=s.read(ROOT/'runs/confirmation/manifest.json');c=manifest['config']
    tasks={t['seed']:t for t in s.read(ROOT/'data/confirmation.json')['tasks']}
    s.initialize(True);rows=[];worst200=0.;worst1000=0.;replayed=0;predictions_checked=0;started=time.perf_counter()
    states=np.arange(0,1001,20);held=independent.HELD
    for ji,job in enumerate(manifest['jobs']):
        s.guard(c,output);name=f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}"
        source=ROOT/'runs/confirmation'/name;task=tasks[job['task_seed']];arch=job['architecture']
        with np.load(source/'base.npz') as z:a0=z['0.weight'];b=z['2.weight']
        with np.load(source/'updates.npz') as z:old=z['predictions'][:,c['arms'].index('freeze_readout'),200,:].copy()
        allpred=[];allweights=[];progress=[]
        for bi,branch in enumerate(['f','g']):
            a=a0.copy();changed=np.arange(8) if bi==0 else np.arange(8,16);stable=np.arange(8,16) if bi==0 else np.arange(8)
            truth=independent.target(task,'40' if bi==0 else '04');base=independent.forward(a,b,arch)
            saved=[];hits={};hitpred={}
            for step in range(1001):
                if step:
                    pre=a[:,changed].T;hidden=independent.activation(pre,arch)
                    err=(hidden@b.T).ravel()-truth[changed]
                    derivative=np.ones_like(pre) if arch=='linear' else np.where(pre>0,1.,.9)
                    a[:,changed]-=.05*((2/8)*err[:,None]*b*derivative).T
                changed_values=(independent.activation(a[:,changed].T,arch)@b.T).ravel()
                ce=float(np.sqrt(np.mean((changed_values-truth[changed])**2)))
                for threshold in [.1,.01]:
                    if ce<=threshold and str(threshold) not in hits:
                        hits[str(threshold)]=step;hitpred[str(threshold)]=independent.forward(a,b,arch)
                if step%20==0:
                    p=independent.forward(a,b,arch);saved.append(p)
                    if step==200:
                        delta=float(np.max(np.abs(p-old[bi])));worst200=max(worst200,delta);assert delta<1e-8
            saved=np.array(saved);allpred.append(saved);allweights.append(a.copy())
            np.testing.assert_array_equal(a[:,stable],a0[:,stable])
            np.testing.assert_array_equal(saved[:,stable],np.broadcast_to(saved[0,stable],(51,8)))
            error=(saved[-1]-truth)**2
            rows.append(dict(job,branch=branch,changed=float(np.sqrt(error[changed].mean())),stable=float(np.sqrt(error[stable].mean())),
                stable_drift=float(np.sqrt(np.mean((saved[-1,stable]-base[stable])**2))),transfer=float(np.sqrt(error[held].mean()/2)),
                first_0_10=hits.get('0.1'),first_0_01=hits.get('0.01')))
            progress.append(dict(steps=hits,predictions={k:v.tolist() for k,v in hitpred.items()}))
            if job['task_seed'] in [91001,91064] and job['model_seed']==2701:
                torch=s.torch;model=amended.network(job['model_seed'],arch,32)
                with torch.no_grad():model[0].weight.copy_(torch.tensor(a0));model[2].weight.copy_(torch.tensor(b))
                model[2].weight.requires_grad_(False);x=torch.tensor(independent.X[changed]);y=torch.tensor(truth[changed])
                optimizer=torch.optim.SGD([model[0].weight],lr=.05)
                for step in range(1000):
                    optimizer.zero_grad(set_to_none=True);loss=((model(x).ravel()-y)**2).mean();loss.backward();optimizer.step()
                endpoint=s.pred(model,torch.tensor(independent.X));delta=float(np.max(np.abs(endpoint-saved[-1])))
                worst1000=max(worst1000,delta);assert delta<1e-8
                np.testing.assert_allclose(model[0].weight.detach().numpy(),a,atol=1e-8,rtol=0)
                np.testing.assert_array_equal(model[2].weight.detach().numpy(),b);replayed+=1
            predictions_checked+=saved.size
        np.savez_compressed(output/f'{name}.npz',predictions=np.array(allpred),steps=states,hidden_weights=np.array(allweights),readout=b,
                            targets=np.stack([independent.target(task,'40'),independent.target(task,'04')]))
        s.write(output/f'{name}.json',dict(job=job,progress=progress))
        if (ji+1)%64==0:print(json.dumps(dict(complete_models=ji+1,total_models=512)),flush=True)
    descriptive={}
    for arch in c['architectures']:
        descriptive[arch]={}
        for history in c['histories']:
            selected=[r for r in rows if r['architecture']==arch and r['history']==history]
            descriptive[arch][history]=dict(trajectories=len(selected),
                endpoint={key:dict(mean=float(np.mean([r[key] for r in selected])),maximum=max(r[key] for r in selected)) for key in ['changed','stable','stable_drift','transfer']},
                progress={key:dict(reached=sum(r[key] is not None for r in selected),total=len(selected),
                    mean_steps=float(np.mean([r[key] for r in selected if r[key] is not None])),
                    maximum_steps=max(r[key] for r in selected if r[key] is not None)) for key in ['first_0_10','first_0_01']})
    s.write(output/'records.json',rows)
    result=dict(status='complete',post_confirmation=True,primary_analysis_changed=False,base_models_reused=512,trajectories=len(rows),
        steps=1000,recorded_steps=states.tolist(),per_example_records=predictions_checked,numpy_vs_original_200_max_error=worst200,
        independent_torch_replayed_trajectories=replayed,torch_vs_numpy_1000_max_error=worst1000,all_intervention_identity_checks='passed',
        descriptive=descriptive,seconds=time.perf_counter()-started,created_utc=s.now(),
        source_hashes={str(p.relative_to(ROOT)):s.sha(p) for p in [Path(__file__),ROOT/'code/audit.py',ROOT/'code/study.py',ROOT/'code/amended_study.py',ROOT/'protocol/DURATION-CONTROL.md']})
    s.write(output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    REPLAY_OUTPUT=parser.parse_args().output
    run()
