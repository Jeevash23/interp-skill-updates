"""Exact fresh-study replay for first/last families, both histories and architectures."""
import argparse
from pathlib import Path
import numpy as np
import study as s
import amended_study as amended


def replay(run,output):
    run,output=map(Path,[run,output]);output.mkdir(parents=True,exist_ok=False)
    m=s.read(run/'manifest.json');c=m['config'];s.initialize(True);s.network=amended.network
    assert s.read(run/'status.json')['status']=='complete'
    tasks=s.read(run/'snapshot/data/confirmation.json')['tasks'];rows,x=s.rows_and_inputs();x=s.torch.tensor(x)
    train=np.array([r['split']=='train' for r in rows]);checked=[];arrays=0
    selected=[j for j in m['jobs'] if j['task_seed'] in [c['task_seeds'][0],c['task_seeds'][-1]] and j['model_seed']==c['initialization_seeds'][0]]
    for j in selected:
        s.guard(c,output)
        name=f"t{j['task_seed']}-s{j['model_seed']}-{j['architecture']}-{j['history']}";dest=output/name;dest.mkdir()
        task=next(t for t in tasks if t['seed']==j['task_seed'])
        model=s.train_base(task,j['model_seed'],j['architecture'],j['history'],c,dest,x,train)
        s.updates(task,j['model_seed'],model,c,dest,x)
        for file in dest.glob('*.npz'):
            with np.load(file) as a,np.load(run/name/file.name) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key]);arrays+=1
        assert s.read(dest/'base.json')['training']==s.read(run/name/'base.json')['training']
        assert s.read(dest/'updates.json')==s.read(run/name/'updates.json')
        checked.append(name)
        print(name+': exact replay passed',flush=True)
    result=dict(status='passed',models=len(checked),trajectories=len(checked)*18,array_comparisons=arrays,
                selected_models=checked,bitwise_identical=True,source_sha256=s.sha(__file__),completed_utc=s.now())
    s.write(output/'verification.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();replay(a.run,a.output)
