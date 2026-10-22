"""Independent audit of the explicit modular reference and its fixed composition."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import audit as a

np,torch=a.np,a.torch


def logsoftmax(x):
    x=x-x.max(-1,keepdims=True)
    return x-np.log(np.exp(x).sum(-1,keepdims=True))


def numpy_reference(state,rows):
    result=[]
    for row in rows:
        if row['query']=='f':
            y=state['f.weight'][:,row['a']].numpy().astype(np.float64)
        elif row['query']=='g':
            y=state['g.weight'][:,row['b']].numpy().astype(np.float64)
        else:
            f=logsoftmax(state['f.weight'][:,row['a']].numpy().astype(np.float64))
            g=logsoftmax(state['g.weight'][:,row['b']].numpy().astype(np.float64))
            y=[]
            for target in range(8):
                terms=np.array([f[i]+g[(target-i)%8] for i in range(8)])
                m=terms.max()
                y.append(m+np.log(np.exp(terms-m).sum()))
        result.append(y)
    return np.array(result)


def audit_reference(run,output):
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest=a.read(run/'manifest.json')
    summary=a.read(run/'summary.json')
    for path,value in manifest['input_hashes'].items():
        assert hashlib.sha256((run/'snapshot'/path).read_bytes()).hexdigest()==value
    tasks={t['task_seed']:t for t in a.read(run/'snapshot/03-Data/prepared/task-families.json')}
    groups=defaultdict(list)
    total=0
    with gzip.open(run/'observations.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            task=tasks[row['task_seed']]
            assert row['input_hashes']==manifest['input_hashes']
            assert row['target_class']==a.label(task,row['environment'],row['query'],row['a'],row['b'])
            assert row['predicted_class']==int(np.argmax(row['logits']))
            assert row['split']==('heldout_pairs' if a.category(row)=='combined_heldout' else 'train')
            groups[(row['task_seed'],row['model_seed'],row['branch'],row['step'])].append(row)
            total+=1
    maximum_error=0.
    checkpoints=0
    for job in summary['models']:
        task=tasks[job['task_seed']]
        folder=run/f"task{job['task_seed']}-init{job['model_seed']}"
        base=torch.load(folder/'base.pt',map_location='cpu',weights_only=True)['model']
        rows=groups[(job['task_seed'],job['model_seed'],'base',200)]
        assert len(rows)==80
        assert a.independent_metrics(rows,task,'E00')==job['base']
        np.testing.assert_allclose(numpy_reference(base,rows),[r['logits'] for r in rows],atol=2e-6,rtol=2e-5)
        checkpoints+=1
        for branch in job['branches']:
            primitive='f' if branch['branch']=='update_f' else 'g'
            stable='g.weight' if primitive=='f' else 'f.weight'
            expected_ids=[r['id'] for r in task['environments'][branch['branch']]['examples'] if r['query']==primitive]
            assert branch['training_example_ids']==expected_ids
            for point in branch['curve']:
                rows=groups[(job['task_seed'],job['model_seed'],branch['branch'],point['step'])]
                assert len(rows)==80
                assert a.independent_metrics(rows,task,branch['branch'])==point['metrics']
            rows=groups[(job['task_seed'],job['model_seed'],branch['branch'],400)]
            end=torch.load(folder/(branch['branch']+'.pt'),map_location='cpu',weights_only=True)['model']
            np.testing.assert_array_equal(base[stable],end[stable])
            expected=numpy_reference(end,rows)
            observed=np.array([r['logits'] for r in rows])
            np.testing.assert_allclose(expected,observed,atol=2e-6,rtol=2e-5)
            np.testing.assert_array_equal(expected.argmax(-1),observed.argmax(-1))
            maximum_error=max(maximum_error,float(np.max(np.abs(expected-observed))))
            checkpoints+=1
    expected_count=len(summary['models'])*80*(1+2*9)
    assert total==expected_count
    good=all(all(v['accuracy']==1. for v in model['base'].values()) and all(all(v['accuracy']==1. for v in branch['curve'][-1]['metrics'].values()) for branch in model['branches']) for model in summary['models'])
    assert good==summary['assay_passed']
    result=dict(status='passed',raw_records_checked=total,checkpoints_checked=checkpoints,
                reference_assay_passed=good,maximum_numpy_absolute_error=maximum_error,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    audit_reference(args.run,args.output)
