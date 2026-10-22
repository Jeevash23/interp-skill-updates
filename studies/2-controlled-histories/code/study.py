"""AI-07 controlled histories, dense additive regressors, and causal update controls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def guard(config,output):
    info=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
    memory=int(info['MemAvailable'].split()[0])*1024
    output=Path(output)
    while not output.exists():output=output.parent
    disk=shutil.disk_usage(output).free
    if memory<config['minimum_memory_gib']*2**30 or disk<config['minimum_disk_gib']*2**30:
        raise RuntimeError('Resource guard: insufficient available memory or disk')
    return dict(available_memory_bytes=memory,free_disk_bytes=disk)


def initialize(framework=True):
    global np,torch
    import numpy as np
    if framework:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True)


def rows_and_inputs():
    rows=[];x=[]
    for q in ['f','g','combine']:
        values=range(8) if q!='combine' else [(a,b) for a in range(8) for b in range(8)]
        for v in values:
            a,b=(v,None) if q=='f' else (None,v) if q=='g' else v
            z=np.zeros(16)
            if a is not None:z[a]=1
            if b is not None:z[8+b]=1
            held=q=='combine' and (b-a)%8 in [0,1]
            rows.append(dict(query=q,a=a,b=b,split='heldout' if held else 'train'))
            x.append(z)
    return rows,np.array(x)


def schedules(config):
    first=['00','20','22','02','11','31','33','13']
    second=['00','11','02','13','22','33','20','31']
    suffix=['00','20','22','02','00']
    return {'independent':first*config['cycles']+suffix,'joint':second*config['cycles']+suffix}


def labels(task,environment):
    fi,gi=map(int,environment)
    f,g=np.array(task['f'][fi]),np.array(task['g'][gi])
    return np.concatenate([f,g,(f[:,None]+g[None,:]).reshape(-1)])


def make_task(seed):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(np.column_stack([np.ones(8),rng.normal(size=(8,3))]))
    fb=q[:,1:4]*np.sqrt(8)
    incidence=np.array([[float((b-a)%8 not in (0,1)) for b in range(8)] for a in range(8)])
    constraints=np.column_stack([np.ones(8),incidence.T@fb[:,0],incidence.T@fb[:,1]])
    u,s,v=np.linalg.svd(constraints,full_matrices=True)
    null=u[:,int((s>1e-10).sum()):]
    gb,_=np.linalg.qr(null@rng.normal(size=(null.shape[1],3)))
    gb=gb*np.sqrt(8)
    return dict(seed=seed,f=np.stack([fb[:,0],fb[:,1],-fb[:,0],-fb[:,1],fb[:,2]]).tolist(),
                g=np.stack([gb[:,0],gb[:,1],-gb[:,0],-gb[:,1],gb[:,2]]).tolist())


def check_design(task,config):
    rows,x=rows_and_inputs();train=np.array([r['split']=='train' for r in rows])
    orders=schedules(config)
    assert Counter(orders['independent'])==Counter(orders['joint'])
    assert orders['independent'][-5:]==orders['joint'][-5:]
    sizes=[];changes={}
    for name,order in orders.items():
        values=[labels(task,env)[train] for env in order]
        assert max(abs(float(np.mean(y*y))-1.75) for y in values)<1e-10
        assert max(abs(float(y.mean())) for y in values)<1e-10
        distance=[float(np.mean((b-a)**2)) for a,b in zip(values,values[1:])]
        assert max(abs(d-3.5) for d in distance)<1e-10
        sizes+=distance
        changes[name]=Counter(sum(a!=b for a,b in zip(old,new)) for old,new in zip(order,order[1:]))
    for key in ['f','g']:
        a=np.array(task[key])
        np.testing.assert_allclose(a.mean(1),0,atol=1e-12)
        np.testing.assert_allclose(np.mean(a*a,axis=1),1,atol=1e-12)
        np.testing.assert_allclose(a[[0,1,4]]@a[[0,1,4]].T,np.eye(3)*8,atol=1e-10)
    return dict(seed=task['seed'],training_examples=int(train.sum()),withheld_examples=int((~train).sum()),
                environment_count=len(orders['joint']),transition_mse_min=min(sizes),transition_mse_max=max(sizes),
                changed_function_counts=changes)


def prepare(config_path,output):
    initialize(False)
    config=read(config_path);output=Path(output)
    tasks=[make_task(s) for s in config['task_seeds']]
    checks=[check_design(t,config) for t in tasks]
    obj=dict(created_utc=now(),configuration_sha256=sha(config_path),tasks=tasks,design_checks=checks,
             schedules=schedules(config),rows=rows_and_inputs()[0],source_sha256=sha(__file__))
    with output.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status='prepared',tasks=len(tasks),design_matching='passed',output=str(output))))


def network(seed,architecture,width):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16,width,bias=False),
                               torch.nn.Identity() if architecture=='linear' else torch.nn.ReLU(),
                               torch.nn.Linear(width,1,bias=False))


def model_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def weights(model):return {name:p.detach().numpy().copy() for name,p in model.state_dict().items()}


def pred(model,x):
    with torch.no_grad():return model(x).reshape(-1).numpy().copy()


def errors(values,target):
    d=(values-target)**2
    rows,_=rows_and_inputs()
    seen=[i for i,r in enumerate(rows) if r['query']=='combine' and r['split']=='train']
    unseen=[i for i,r in enumerate(rows) if r['split']=='heldout']
    return dict(f=float(np.sqrt(d[:8].mean())),g=float(np.sqrt(d[8:16].mean())),
                combined_seen=float(np.sqrt(d[seen].mean()/2)),combined_unseen=float(np.sqrt(d[unseen].mean()/2)))


def train_base(task,seed,architecture,history,config,folder,x,train):
    model=network(seed,architecture,config['width'])
    initial=model_hash(model)
    np.savez_compressed(folder/'initial.npz',**weights(model))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['base_learning_rate'])
    trace=[];order=schedules(config)[history]
    previous=None;started=time.perf_counter()
    for block,env in enumerate(order):
        target=labels(task,env);y=torch.tensor(target[train]);xb=x[train]
        with torch.no_grad():before=float(((model(xb).reshape(-1)-y)**2).mean())
        oldloss=None if previous is None else float(np.mean((pred(model,x)[train]-previous[train])**2))
        steps=config['steps_per_environment']+(config['final_extra_steps'] if block==len(order)-1 else 0)
        first_gradient=None
        for s in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss=((model(xb).reshape(-1)-y)**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite base loss')
            loss.backward()
            if s==0:first_gradient=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
            optimizer.step()
        prediction=pred(model,x)
        np.savez_compressed(folder/f'environment-{block:02d}.npz',**weights(model),predictions=prediction,targets=target)
        trace.append(dict(block=block,environment=env,steps=steps,loss_before=before,loss_previous=oldloss,
                          loss_jump=None if oldloss is None else before-oldloss,initial_gradient_norm=first_gradient,
                          errors=errors(prediction,target)))
        previous=target
    np.savez_compressed(folder/'base.npz',**weights(model),predictions=pred(model,x),targets=labels(task,'00'))
    torch.save(optimizer.state_dict(),folder/'base-optimizer.pt')
    write(folder/'base.json',dict(initial_hash=initial,final_hash=model_hash(model),training=trace,
                                 seconds=time.perf_counter()-started,errors=trace[-1]['errors']))
    return model


def mask_for(arm,task_seed,model_seed,branch,width):
    hidden=np.ones((width,16));readout=np.ones((1,width))
    if arm=='freeze_readout':readout[:]=0
    if arm=='readout_only':hidden[:]=0
    if arm.startswith('random_freeze_'):
        draw=int(arm.rsplit('_',1)[1]);rng=np.random.default_rng(task_seed*100000+model_seed*10+draw+(0 if branch=='f' else 70000000))
        cols=range(8) if branch=='f' else range(8,16)
        active=[r*16+c for r in range(width) for c in cols]+list(range(width*16,width*17))
        chosen=rng.choice(active,size=width,replace=False)
        joined=np.concatenate([hidden.reshape(-1),readout.reshape(-1)]);joined[chosen]=0
        hidden=joined[:width*16].reshape(width,16);readout=joined[width*16:].reshape(1,width)
    return [torch.tensor(hidden),torch.tensor(readout)]


def updates(task,seed,base,config,folder,x):
    steps=config['update_steps'];arms=config['arms']
    baseline=pred(base,x);base_flat=np.concatenate([p.detach().numpy().reshape(-1) for p in base.parameters()])
    predictions=np.empty((2,len(arms),steps+1,80));deltas=np.empty((2,len(arms),steps+1));paths=np.empty_like(deltas)
    drifts=np.empty((2,len(arms),steps+1,8));norms=np.zeros_like(deltas)
    endweights={};information=[];base_hash=model_hash(base)
    for bi,branch in enumerate(['f','g']):
        target=labels(task,'40' if branch=='f' else '04')
        ids=np.arange(8) if branch=='f' else np.arange(8,16);stable=np.arange(8,16) if branch=='f' else np.arange(8)
        with torch.no_grad():hidden_stable=base[1](base[0](x[stable])).numpy().copy()
        for ai,arm in enumerate(arms):
            model=copy.deepcopy(base);masks=mask_for(arm,task['seed'],seed,branch,config['width'])
            y=torch.tensor((labels(task,'00') if arm=='old_primitive' else target)[ids]);xb=x[ids]
            path=0.;max_drift_error=0.;zeros=0
            for s in range(steps+1):
                if s and arm!='no_update':
                    model.zero_grad(set_to_none=True)
                    loss=((model(xb).reshape(-1)-y)**2).mean();loss.backward()
                    gradients=[p.grad*m for p,m in zip(model.parameters(),masks)]
                    norm=float(sum(g.square().sum() for g in gradients).sqrt())
                    scale=config['update_learning_rate']
                    if arm=='normalized':
                        if norm>1e-14:scale=config['normalized_step_size']/norm
                        else:scale=0.;zeros+=1
                    with torch.no_grad():
                        for p,g in zip(model.parameters(),gradients):p.add_(g,alpha=-scale)
                    path+=scale*norm;norms[bi,ai,s]=scale*norm
                values=pred(model,x)
                if not np.isfinite(values).all():raise FloatingPointError('Nonfinite adaptation output')
                predictions[bi,ai,s]=values
                current=np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])
                deltas[bi,ai,s]=np.linalg.norm(current-base_flat);paths[bi,ai,s]=path
                change=model[2].weight.detach().numpy().reshape(-1)-base[2].weight.detach().numpy().reshape(-1)
                predicted_drift=hidden_stable@change
                drifts[bi,ai,s]=predicted_drift
                max_drift_error=max(max_drift_error,float(np.max(np.abs((values[stable]-baseline[stable])-predicted_drift))))
            cols=slice(8,16) if branch=='f' else slice(0,8)
            assert torch.equal(model[0].weight[:,cols],base[0].weight[:,cols]),'Untargeted input columns changed'
            if arm=='freeze_readout':assert torch.equal(model[2].weight,base[2].weight)
            if arm=='no_update':assert model_hash(model)==base_hash
            assert max_drift_error<1e-9,'Readout mediation identity failed'
            for name,value in weights(model).items():endweights[f'{branch}__{arm}__{name}']=value
            information.append(dict(branch=branch,arm=arm,readout_identity_max_error=max_drift_error,
                                    zero_gradient_steps=zeros,final_errors=errors(predictions[bi,ai,-1],target),
                                    final_parameter_distance=float(deltas[bi,ai,-1]),path_length=float(paths[bi,ai,-1]),
                                    frozen_weights=int(sum((m==0).sum() for m in masks))))
            assert model_hash(base)==base_hash
    np.savez_compressed(folder/'updates.npz',predictions=predictions,parameter_distance=deltas,path_length=paths,
                        step_length=norms,readout_predicted_drift=drifts,targets=np.stack([labels(task,'40'),labels(task,'04')]))
    np.savez_compressed(folder/'endpoints.npz',**endweights)
    write(folder/'updates.json',dict(branches=['f','g'],arms=arms,steps=list(range(steps+1)),trajectories=information,
                                    rows=rows_and_inputs()[0],normalization='primitive RMS 1; combined RMS sqrt(2)'))


def execute(config_path,tasks_path,output):
    config_path,tasks_path,output=map(lambda p:Path(p).resolve(),[config_path,tasks_path,output])
    config=read(config_path);resources=guard(config,output.parent)
    output.mkdir(parents=True,exist_ok=False);initialize(True)
    bundle=read(tasks_path);tasks=bundle['tasks']
    assert bundle['configuration_sha256']==sha(config_path)
    assert [t['seed'] for t in tasks]==config['task_seeds']
    rows,values=rows_and_inputs();x=torch.tensor(values);train=np.array([r['split']=='train' for r in rows])
    for task in tasks:check_design(task,config)
    sources=[Path(__file__).resolve(),config_path,tasks_path,ROOT/'protocol/DEVELOPMENT.md']
    if (ROOT/'protocol/CONFIRMATION.md').exists():sources.append(ROOT/'protocol/CONFIRMATION.md')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        dest=output/'snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    jobs=[dict(task_seed=t['seed'],model_seed=seed,architecture=arch,history=h) for t in tasks for seed in config['initialization_seeds'] for arch in config['architectures'] for h in config['histories']]
    write(output/'manifest.json',dict(stage=config['stage'],created_utc=now(),config=config,source_hashes=hashes,jobs=jobs,
                                      resources=resources,environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,torch=torch.__version__,dtype='float64',threads=1,gpu_used=False)))
    started=time.perf_counter();done=[];failed=[]
    for job in jobs:
        jid=f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}"
        folder=output/jid;folder.mkdir();begin=time.perf_counter()
        try:
            guard(config,output);task=next(t for t in tasks if t['seed']==job['task_seed'])
            model=train_base(task,job['model_seed'],job['architecture'],job['history'],config,folder,x,train)
            updates(task,job['model_seed'],model,config,folder,x)
            record=dict(status='complete',job=job,seconds=time.perf_counter()-begin,finished_utc=now())
            write(folder/'status.json',record);done.append(jid)
            print(json.dumps(dict(job=jid,status='complete',seconds=round(record['seconds'],2),done=len(done),total=len(jobs))),flush=True)
            del model
        except Exception as exc:
            record=dict(status='failed',job=job,error=str(exc),traceback=traceback.format_exc(),utc=now())
            write(folder/'status.json',record);failed.append(record);print(json.dumps(record),flush=True)
            if isinstance(exc,(MemoryError,RuntimeError)):break
        write(output/'status.json',dict(status='running',completed=done,failures=failed,planned=len(jobs),elapsed_seconds=time.perf_counter()-started))
    final=dict(status='complete' if len(done)==len(jobs) and not failed else 'incomplete',completed=done,failures=failed,
               planned=len(jobs),seconds=time.perf_counter()-started,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=now())
    write(output/'status.json',final);print(json.dumps(final),flush=True)
    if final['status']!='complete':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--tasks',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.config,a.output)
    else:execute(a.config,a.tasks,a.output)
