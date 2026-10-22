"""Check every saved duration-control endpoint with PyTorch and reconstruct summaries."""
import json
from pathlib import Path
import numpy as np
import study as s
import amended_study as amended

ROOT=Path(__file__).resolve().parents[1]


def main(output):
    s.initialize(True);run=ROOT/'runs/duration-control';summary=s.read(run/'summary.json');records=s.read(run/'records.json')
    lookup={(r['task_seed'],r['model_seed'],r['architecture'],r['history'],r['branch']):r for r in records}
    manifest=s.read(ROOT/'runs/confirmation/manifest.json');tasks={t['seed']:t for t in s.read(ROOT/'data/confirmation.json')['tasks']}
    rows,values=s.rows_and_inputs();x=s.torch.tensor(values);held=[i for i,r in enumerate(rows) if r['split']=='heldout']
    count=0;worst=0.;allrecords=0
    for job in manifest['jobs']:
        name=f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}"
        with np.load(run/f'{name}.npz') as z,np.load(ROOT/'runs/confirmation'/name/'base.npz') as base:
            p=z['predictions'];meta=s.read(run/f'{name}.json')
            assert p.shape==(2,51,80);np.testing.assert_array_equal(z['steps'],np.arange(0,1001,20))
            np.testing.assert_array_equal(z['readout'],base['2.weight'])
            for bi,branch in enumerate(['f','g']):
                model=amended.network(job['model_seed'],job['architecture'],32)
                with s.torch.no_grad():model[0].weight.copy_(s.torch.tensor(z['hidden_weights'][bi]));model[2].weight.copy_(s.torch.tensor(z['readout']))
                delta=float(np.max(np.abs(s.pred(model,x)-p[bi,-1])));worst=max(worst,delta);assert delta<1e-9
                truth=s.labels(tasks[job['task_seed']],'40' if bi==0 else '04');np.testing.assert_array_equal(truth,z['targets'][bi])
                changed=np.arange(8) if bi==0 else np.arange(8,16);stable=np.arange(8,16) if bi==0 else np.arange(8)
                np.testing.assert_array_equal(z['hidden_weights'][bi][:,stable],base['0.weight'][:,stable])
                np.testing.assert_array_equal(p[bi][:,stable],np.broadcast_to(p[bi,0,stable],(51,8)))
                r=lookup[(job['task_seed'],job['model_seed'],job['architecture'],job['history'],branch)]
                for key,ix,scale in [('changed',changed,1),('stable',stable,1),('transfer',held,2)]:
                    value=np.sqrt(np.mean((p[bi,-1,ix]-truth[ix])**2)/scale);assert abs(value-r[key])<1e-10
                for threshold,key in [(.1,'first_0_10'),(.01,'first_0_01')]:
                    step=meta['progress'][bi]['steps'].get(str(threshold));assert step==r[key]
                    if step is not None:
                        q=np.array(meta['progress'][bi]['predictions'][str(threshold)])
                        assert np.sqrt(np.mean((q[changed]-truth[changed])**2))<=threshold
                count+=1;allrecords+=p[bi].size
    for arch in ['linear','leaky']:
        for history in ['independent','joint']:
            rr=[r for r in records if r['architecture']==arch and r['history']==history];reported=summary['descriptive'][arch][history]
            assert len(rr)==reported['trajectories']==256
            for metric in ['changed','stable','transfer','stable_drift']:
                np.testing.assert_allclose([reported['endpoint'][metric]['mean'],reported['endpoint'][metric]['maximum']],
                    [np.mean([r[metric] for r in rr]),max(r[metric] for r in rr)],atol=1e-10,rtol=0)
    assert count==1024 and allrecords==summary['per_example_records']
    result=dict(status='passed',trajectories=count,endpoint_checkpoints=count,per_example_records=allrecords,
        torch_forward_max_error=worst,fixed_readout_and_stable_predictions='passed',summary_reconstruction='passed',
        source_sha256=s.sha(__file__),created_utc=s.now())
    s.write(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
