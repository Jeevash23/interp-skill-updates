"""Independent NumPy reconstruction; does not import the experiment or analysis."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.stats import t


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target(task,env):
    f=np.array(task['f'][int(env[0])]);g=np.array(task['g'][int(env[1])])
    return np.r_[f,g,[f[a]+g[b] for a in range(8) for b in range(8)]]


X=np.r_[np.c_[np.eye(8),np.zeros((8,8))],np.c_[np.zeros((8,8)),np.eye(8)],
        np.array([np.r_[np.eye(8)[a],np.eye(8)[b]] for a in range(8) for b in range(8)])]
HELD=np.array([16+a*8+b for a in range(8) for b in range(8) if (b-a)%8 in (0,1)])
TRAIN=np.array([i for i in range(80) if i not in HELD])
SEEN=np.array([i for i in TRAIN if i>=16])


def activation(x,arch):
    return x if arch=='linear' else np.maximum(0,x) if arch=='relu' else np.where(x>0,x,.9*x)


def forward(a,b,arch,x=X):return (activation(x@a.T,arch)@b.T).ravel()


def masks(arm,job,branch):
    a=np.ones((32,16));b=np.ones((1,32))
    if arm=='freeze_readout':b[:]=0
    if arm=='readout_only':a[:]=0
    if arm.startswith('random_freeze_'):
        seed=job['task_seed']*100000+job['model_seed']*10+int(arm[-1])+(70000000 if branch=='g' else 0)
        columns=range(8) if branch=='f' else range(8,16)
        choices=[16*r+c for r in range(32) for c in columns]+list(range(512,544))
        frozen=np.random.default_rng(seed).choice(choices,32,replace=False)
        for idx in frozen:
            if idx<512:a.flat[idx]=0
            else:b.flat[idx-512]=0
    return a,b


def replay(base_a,base_b,job,branch,arm,truth,recorded,c):
    a=base_a.copy();b=base_b.copy();arch=job['architecture']
    ids=np.arange(8) if branch=='f' else np.arange(8,16);x=X[ids];ma,mb=masks(arm,job,branch)
    worst=0.;path=0.
    for step in range(c['update_steps']+1):
        if step and arm!='no_update':
            h0=x@a.T;h=activation(h0,arch);err=(h@b.T).ravel()-truth[ids]
            gb=(2/8)*(err@h)[None,:]
            derivative=np.ones_like(h0) if arch=='linear' else (h0>0).astype(float) if arch=='relu' else np.where(h0>0,1.,.9)
            ga=((2/8)*err[:,None]*b*derivative).T@x
            ga*=ma;gb*=mb;norm=np.sqrt(np.sum(ga*ga)+np.sum(gb*gb))
            rate=c['update_learning_rate'] if arm!='normalized' else c['normalized_step_size']/norm if norm>1e-14 else 0.
            a-=rate*ga;b-=rate*gb;path+=rate*norm
        worst=max(worst,float(np.max(np.abs(forward(a,b,arch)-recorded[step]))))
    assert worst<1e-8,(job,branch,arm,worst)
    return worst


def audit(run,analysis,output):
    run,analysis,output=map(Path,[run,analysis,output]);m=read(run/'manifest.json');c=m['config'];status=read(run/'status.json')
    assert status['status']=='complete' and len(status['completed'])==len(m['jobs']) and not status['failures']
    for name,sha in m['source_hashes'].items():assert digest(run/'snapshot'/name)==sha
    bundle=read(run/'snapshot/data'/('confirmation.json' if c['stage']=='confirmation' else 'amendment.json' if c['stage']=='nonlinear_diagnostic' else 'development.json'))
    tasks={a['seed']:a for a in bundle['tasks']};assert sorted(tasks)==c['task_seeds']
    orders=bundle['schedules'];assert Counter(orders['independent'])==Counter(orders['joint'])
    assert orders['independent'][-5:]==orders['joint'][-5:]
    for task in tasks.values():
        for order in orders.values():
            ys=np.array([target(task,e)[TRAIN] for e in order])
            np.testing.assert_allclose(np.mean(ys,axis=1),0,atol=1e-10)
            np.testing.assert_allclose(np.mean(ys**2,axis=1),1.75,atol=1e-10)
            np.testing.assert_allclose(np.mean(np.diff(ys,axis=0)**2,axis=1),3.5,atol=1e-10)
    records=read(analysis/'records.json');lookup={(r['task_seed'],r['model_seed'],r['architecture'],r['history'],r['branch'],r['arm']):r for r in records}
    scalar_errors=[];initials={};ncheck=0;nrecords=0;worst=0.;replay_worst=0.;replays=0;families=defaultdict(list)
    for job in m['jobs']:
        folder=run/f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}"
        task=tasks[job['task_seed']];arch=job['architecture'];init=np.load(folder/'initial.npz')
        initkey=(job['task_seed'],job['model_seed'],arch)
        current=np.r_[init['0.weight'].ravel(),init['2.weight'].ravel()]
        if initkey in initials:np.testing.assert_array_equal(initials[initkey],current)
        else:initials[initkey]=current
        for block,env in enumerate(orders[job['history']]):
            with np.load(folder/f'environment-{block:02d}.npz') as z:
                np.testing.assert_array_equal(z['targets'],target(task,env))
                e=float(np.max(np.abs(forward(z['0.weight'],z['2.weight'],arch)-z['predictions'])))
                worst=max(worst,e);assert e<1e-9;ncheck+=1
        base=np.load(folder/'base.npz');a=base['0.weight'];b=base['2.weight'];bp=forward(a,b,arch)
        ncheck+=1
        assert a.shape==(32,16) and b.shape==(1,32)
        np.testing.assert_array_equal(base['targets'],target(task,'00'));np.testing.assert_allclose(bp,base['predictions'],atol=1e-9,rtol=0)
        bm=read(folder/'base.json');diff=(bp-base['targets'])**2
        for name,indices,scale in [('f',np.arange(8),1),('g',np.arange(8,16),1),('combined_seen',SEEN,2),('combined_unseen',HELD,2)]:
            assert abs(np.sqrt(diff[indices].mean()/scale)-bm['errors'][name])<1e-9
        endpoints=np.load(folder/'endpoints.npz')
        with np.load(folder/'updates.npz') as z:
            for bi,branch in enumerate(['f','g']):
                changed=np.arange(8) if branch=='f' else np.arange(8,16);stable=np.arange(8,16) if branch=='f' else np.arange(8)
                truth=target(task,'40' if branch=='f' else '04');np.testing.assert_array_equal(truth,z['targets'][bi])
                hs=activation(X[stable]@a.T,arch)
                for ai,arm in enumerate(c['arms']):
                    pa=endpoints[f'{branch}__{arm}__0.weight'];pb=endpoints[f'{branch}__{arm}__2.weight'];p=z['predictions'][bi,ai]
                    np.testing.assert_allclose(p[0],bp,atol=1e-9,rtol=0)
                    end=forward(pa,pb,arch);e=float(np.max(np.abs(end-p[-1])));worst=max(worst,e);assert e<1e-9
                    np.testing.assert_array_equal(pa[:,stable],a[:,stable])
                    ma,mb=masks(arm,job,branch)
                    np.testing.assert_array_equal(pa[ma==0],a[ma==0]);np.testing.assert_array_equal(pb[mb==0],b[mb==0])
                    predicted=(hs@(pb-b).T).ravel()
                    np.testing.assert_allclose(end[stable]-bp[stable],predicted,atol=1e-9,rtol=0)
                    np.testing.assert_allclose(p[:,stable]-p[0,stable],z['readout_predicted_drift'][bi,ai],atol=1e-9,rtol=0)
                    dist=np.sqrt(np.sum((pa-a)**2)+np.sum((pb-b)**2))
                    assert abs(dist-z['parameter_distance'][bi,ai,-1])<1e-9
                    np.testing.assert_allclose(np.cumsum(z['step_length'][bi,ai]),z['path_length'][bi,ai],atol=1e-9,rtol=0)
                    if arm=='normalized':np.testing.assert_allclose(z['step_length'][bi,ai,1:],.01,atol=1e-12,rtol=0)
                    if arm in ['freeze_readout','no_update']:np.testing.assert_allclose(p[:,stable],np.broadcast_to(bp[stable],(c['update_steps']+1,8)),atol=1e-9,rtol=0)
                    if arm=='no_update':np.testing.assert_array_equal(p,np.broadcast_to(p[0],p.shape))
                    ce=np.sqrt(np.mean((p[:,changed]-truth[changed])**2,axis=1));se=np.sqrt(np.mean((p[:,stable]-truth[stable])**2,axis=1));te=np.sqrt(np.mean((p[:,HELD]-truth[HELD])**2,axis=1)/2)
                    r=lookup[(job['task_seed'],job['model_seed'],arch,job['history'],branch,arm)]
                    for name,value in [('changed',ce[-1]),('stable',se[-1]),('transfer',te[-1])]:
                        scalar_errors.append(abs(value-r[name]));assert abs(value-r[name])<1e-10
                        if arm=='unrestricted':families[(arch,name,job['task_seed'],job['history'])].append(float(value))
                    hits=np.flatnonzero(ce<=.1);assert r['progress_reached']==bool(len(hits))
                    assert r['progress_step']==(int(hits[0]) if len(hits) else None)
                    # All 18 arms in a fixed first/last-family sample, both architectures and histories.
                    if job['task_seed'] in [c['task_seeds'][0],c['task_seeds'][-1]] and job['model_seed']==c['initialization_seeds'][0]:
                        tr=target(task,'00') if arm=='old_primitive' else truth
                        replay_worst=max(replay_worst,replay(a,b,job,branch,arm,tr,p,c));replays+=1
                    ncheck+=1;nrecords+=p.size
        base.close();endpoints.close();init.close()
    summary=read(analysis/'summary.json');alpha=.05/(3*len(c['architectures']))
    for arch in c['architectures']:
        for metric in ['changed','stable','transfer']:
            d=np.array([np.mean(families[(arch,metric,seed,'independent')])-np.mean(families[(arch,metric,seed,'joint')]) for seed in c['task_seeds']])
            half=t.ppf(1-alpha/2,len(d)-1)*np.std(d,ddof=1)/np.sqrt(len(d));est=summary['primary'][arch]['effects'][metric]
            np.testing.assert_allclose([est['mean'],est['low'],est['high']],[d.mean(),d.mean()-half,d.mean()+half],atol=1e-10,rtol=0)
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),models=len(m['jobs']),
        checkpoints_verified=ncheck,per_example_trajectory_records=nrecords,
        numpy_forward_max_error=worst,summary_scalar_max_error=max(scalar_errors),
        independent_numpy_trajectories_replayed=replays,numpy_replay_max_error=replay_worst,
        source_snapshot_integrity='passed',matching_and_labels='passed',pairing_and_interventions='passed',
        family_aggregation_and_primary_intervals='passed',failures=[],audit_source_sha256=digest(__file__))
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.analysis,a.output)
