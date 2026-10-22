"""An explicitly factorized positive reference, not an emergent-modularity claim."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

import experiment as e


def run(output):
    output=Path(output).resolve()
    cfg=e.read(e.ROOT/'03-Data/config/execution-v1.json')
    e.resource_guard(cfg,output.parent)
    output.mkdir(parents=True,exist_ok=False)
    e.initialize()
    torch,np=e.torch,e.np

    class ModularReference(torch.nn.Module):
        def __init__(self,seed):
            super().__init__()
            torch.manual_seed(seed)
            self.f=torch.nn.Linear(8,8,bias=False)
            self.g=torch.nn.Linear(8,8,bias=False)
        def forward(self,x):
            f=self.f(x[:,3:11]); g=self.g(x[:,11:19])
            pf=f.log_softmax(-1);pg=g.log_softmax(-1)
            i=torch.arange(8)
            combined=torch.stack([torch.logsumexp(pf+pg[:,(k-i)%8],dim=-1) for k in range(8)],dim=-1)
            return torch.where(x[:,0:1]>0,f,torch.where(x[:,1:2]>0,g,combined))

    tasks=e.read(e.ROOT/'03-Data/prepared/task-families.json')
    steps=[0,1,5,10,25,50,100,200,400]
    settings=dict(architecture='two independent linear lookup maps and fixed categorical circular convolution',
                  base_steps=200,adaptation_steps=400,learning_rate=.03,initialization_seeds=cfg['initialization_seeds'],
                  task_seeds=cfg['task_seeds'],device='cpu',threads=1,classification='post-pilot assay reference')
    files=[Path(__file__).resolve(),e.ROOT/'02-Code/experiment.py',e.ROOT/'06-Study-History/DIAGNOSTIC-2026-10-08.md',
           e.ROOT/'03-Data/prepared/task-families.json']
    hashes={str(p.relative_to(e.ROOT)):e.sha(p) for p in files}
    e.write(output/'manifest.json',dict(created_utc=e.now(),settings=settings,input_hashes=hashes,torch=torch.__version__,numpy=np.__version__))
    for p in files:
        dest=output/'snapshot'/p.relative_to(e.ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    began=time.perf_counter()
    summaries=[]
    with gzip.open(output/'observations.jsonl.gz','wt') as handle:
        for task in tasks:
            if task['task_seed'] not in cfg['task_seeds']: continue
            for seed in cfg['initialization_seeds']:
                job=f"task{task['task_seed']}-init{seed}"
                folder=output/job;folder.mkdir()
                model=ModularReference(seed)
                optimizer=torch.optim.Adam(model.parameters(),lr=.03)
                rows=task['environments']['E00']['examples']
                primitives=[r for r in rows if r['query'] in ('f','g')]
                x,y=e.encode(primitives)
                for _ in range(200):e.step(model,optimizer,x,y)
                common=dict(run_id=output.name,task_seed=task['task_seed'],model_seed=seed,history='explicit_modular_reference',input_hashes=hashes)
                logits=e.predict(model,rows)
                base=e.metrics(logits,rows)
                e.emit(handle,dict(common,environment='E00',branch='base',condition='primitive_training',optimizer_mode='reset'),rows,logits,200,time.perf_counter()-began)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),folder/'base.pt')
                branches=[]
                for branch in ('update_f','update_g'):
                    clone=copy.deepcopy(model)
                    changed,stable=(clone.f,clone.g) if branch=='update_f' else (clone.g,clone.f)
                    stable_before=stable.weight.detach().clone()
                    opt=torch.optim.Adam(changed.parameters(),lr=.03)
                    batch=e.update_batch(task,branch,'changed_primitive')
                    x,y=e.encode(batch)
                    evalrows=task['environments'][branch]['examples']
                    curve=[]
                    for s in range(401):
                        if s>0:e.step(clone,opt,x,y)
                        if s in steps:
                            values=e.predict(clone,evalrows)
                            curve.append(dict(step=s,metrics=e.metrics(values,evalrows)))
                            e.emit(handle,dict(common,environment=branch,branch=branch,condition='changed_primitive',optimizer_mode='reset'),evalrows,values,s,time.perf_counter()-began)
                    assert torch.equal(stable.weight.detach(),stable_before), 'Untargeted map changed'
                    torch.save(dict(model=clone.state_dict(),optimizer=opt.state_dict()),folder/f'{branch}.pt')
                    branches.append(dict(branch=branch,stable_map_bitwise_preserved=True,training_example_ids=[r['id'] for r in batch],curve=curve))
                summaries.append(dict(task_seed=task['task_seed'],model_seed=seed,base=base,branches=branches))
                print(json.dumps(dict(reference_job=job,status='complete')),flush=True)
    good=all(all(m['accuracy']==1. for m in r['base'].values()) and all(all(m['accuracy']==1. for m in b['curve'][-1]['metrics'].values()) for b in r['branches']) for r in summaries)
    e.write(output/'summary.json',dict(status='complete',assay_passed=good,models=summaries,elapsed_seconds=time.perf_counter()-began,
                                      maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(status='complete',assay_passed=good,base_models=len(summaries),update_branches=2*len(summaries))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
