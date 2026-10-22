"""Prespecified family-level analysis of the controlled AI-07 continuation."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
import study as s
s.initialize(False)


def interval(values,alpha=.05):
    a=np.asarray(values,dtype=float);n=len(a)
    mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else 0.
    half=float(t.ppf(1-alpha/2,n-1)*sd/np.sqrt(n)) if n>1 else 0.
    return dict(n=n,mean=mean,sd=sd,low=mean-half,high=mean+half,confidence=1-alpha)


def qualified(errors,c):
    return (max(errors['f'],errors['g'])<=c['base_primitive_nrmse_max'] and
            errors['combined_seen']<=c['base_seen_nrmse_max'] and
            errors['combined_unseen']<=c['base_unseen_nrmse_max'])


def group_family(records,key,metric):
    d=defaultdict(list)
    for r in records:
        if all(r[k]==v for k,v in key.items()):d[r['task_seed']].append(r[metric])
    return {k:float(np.mean(v)) for k,v in sorted(d.items())}


def paired(records,arch,arm,metric,alpha=.05):
    a=group_family(records,dict(architecture=arch,history='independent',arm=arm),metric)
    b=group_family(records,dict(architecture=arch,history='joint',arm=arm),metric)
    assert set(a)==set(b)
    delta={k:a[k]-b[k] for k in a}
    return dict(**interval(list(delta.values()),alpha),family_differences=delta,
                independent=interval(list(a.values())),joint=interval(list(b.values())))


def analyze(run,output):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=s.read(run/'manifest.json');status=s.read(run/'status.json');c=manifest['config']
    assert status['status']=='complete' and not status['failures']
    rows,_=s.rows_and_inputs();held=np.array([i for i,r in enumerate(rows) if r['split']=='heldout'])
    records=[];base=[];curves=defaultdict(list);identity=0.;zero=0
    for job in manifest['jobs']:
        jid=f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}";folder=run/jid
        bm=s.read(folder/'base.json');meta=s.read(folder/'updates.json')
        b=dict(job,qualified=qualified(bm['errors'],c),errors=bm['errors'],initial_hash=bm['initial_hash'])
        base.append(b)
        with np.load(folder/'updates.npz') as z:
            p=z['predictions'];targets=z['targets']
            for bi,branch in enumerate(['f','g']):
                changed=np.arange(8) if branch=='f' else np.arange(8,16)
                stable=np.arange(8,16) if branch=='f' else np.arange(8)
                for ai,arm in enumerate(c['arms']):
                    values=p[bi,ai];squared=(values-targets[bi])**2
                    ce=np.sqrt(squared[:,changed].mean(1));se=np.sqrt(squared[:,stable].mean(1))
                    te=np.sqrt(squared[:,held].mean(1)/2)
                    drift=np.sqrt(((values[:,stable]-values[0,stable])**2).mean(1))
                    reached=np.flatnonzero(ce<=.1);first=int(reached[0]) if len(reached) else None
                    info=next(q for q in meta['trajectories'] if q['branch']==branch and q['arm']==arm)
                    identity=max(identity,info['readout_identity_max_error']);zero+=info['zero_gradient_steps']
                    records.append(dict(job,branch=branch,arm=arm,changed=float(ce[-1]),stable=float(se[-1]),
                        transfer=float(te[-1]),stable_drift=float(drift[-1]),base_stable=float(se[0]),
                        parameter_distance=float(z['parameter_distance'][bi,ai,-1]),path=float(z['path_length'][bi,ai,-1]),
                        progress_reached=bool(len(reached)),progress_step=first,
                        progress_stable=None if first is None else float(se[first]),
                        progress_transfer=None if first is None else float(te[first]),
                        progress_path=None if first is None else float(z['path_length'][bi,ai,first]),
                        readout_identity_error=info['readout_identity_max_error']))
                    curves[(job['architecture'],job['history'],arm)].append(np.stack([ce,se,te,drift]))
    base_results={};qualified_arch=[]
    for arch in c['architectures']:
        ar=[b for b in base if b['architecture']==arch];pairs=[]
        for task in c['task_seeds']:
            for seed in c['initialization_seeds']:
                pair=[b for b in ar if b['task_seed']==task and b['model_seed']==seed]
                assert len(pair)==2 and pair[0]['initial_hash']==pair[1]['initial_hash']
                pairs.append(all(b['qualified'] for b in pair))
        base_results[arch]=dict(qualified_pairs=sum(pairs),total_pairs=len(pairs),qualified_models=sum(b['qualified'] for b in ar),
            models=len(ar),errors={k:dict(mean=float(np.mean([b['errors'][k] for b in ar])),maximum=max(b['errors'][k] for b in ar))
                                 for k in ['f','g','combined_seen','combined_unseen']})
        if sum(pairs)>=c['minimum_qualified_development_pairs']:qualified_arch.append(arch)
    primary={};summaries={};controls={};progress={}
    alpha=.05/(3*len(c['architectures']))
    for arch in c['architectures']:
        effects={metric:paired(records,arch,'unrestricted',metric,alpha) for metric in ['stable','changed','transfer']}
        stable,changed,transfer=[effects[m] for m in ['stable','changed','transfer']]
        competence=base_results[arch]['qualified_pairs']/base_results[arch]['total_pairs']>=.9
        if not competence:decision='base_competence_failed'
        elif stable['high'] < -.05 and changed['high']<.02 and transfer['high']<.02:decision='practically_useful_independent_history_benefit'
        elif stable['low']>-.05 and stable['high']<.05:decision='preservation_equivalent_within_0.05'
        elif stable['low']>.05:decision='independent_history_practically_worse'
        elif stable['low']>-.05:decision='practical_independent_history_benefit_ruled_out'
        else:decision='inconclusive'
        primary[arch]=dict(effects=effects,decision=decision,base_competence_met=competence)
        summaries[arch]={};controls[arch]={};progress[arch]={}
        for history in c['histories']:
            summaries[arch][history]={};controls[arch][history]={};progress[arch][history]={}
            for arm in c['arms']:
                key=dict(architecture=arch,history=history,arm=arm)
                summaries[arch][history][arm]={metric:interval(list(group_family(records,key,metric).values()))
                    for metric in ['changed','stable','transfer','stable_drift','parameter_distance','path']}
                eligible=[r for r in records if all(r[k]==v for k,v in key.items())]
                reached=[r for r in eligible if r['progress_reached']]
                progress[arch][history][arm]=dict(reached=len(reached),total=len(eligible),
                    step_mean=float(np.mean([r['progress_step'] for r in reached])) if reached else None,
                    stable_mean=float(np.mean([r['progress_stable'] for r in reached])) if reached else None,
                    transfer_mean=float(np.mean([r['progress_transfer'] for r in reached])) if reached else None)
            # These secondary intervals are descriptive 95% intervals, without multiplicity correction.
            full=group_family(records,dict(architecture=arch,history=history,arm='unrestricted'),'stable_drift')
            freeze=group_family(records,dict(architecture=arch,history=history,arm='freeze_readout'),'stable_drift')
            random=[group_family(records,dict(architecture=arch,history=history,arm=f'random_freeze_{i}'),'stable_drift') for i in range(3)]
            controls[arch][history]=dict(freeze_minus_unrestricted=interval([freeze[k]-full[k] for k in full]),
                freeze_minus_mean_random=interval([freeze[k]-np.mean([r[k] for r in random]) for k in full]))
    normalized={arch:{metric:paired(records,arch,'normalized',metric) for metric in ['stable','changed','transfer','path']} for arch in c['architectures']}
    # Development determines only competence and the fixed confirmation sample size.
    sd=max([primary[a]['effects']['stable']['sd'] for a in qualified_arch],default=0.)
    proposed=max(c['confirmation_minimum_families'],min(c['confirmation_maximum_families'],math.ceil((2.3*sd/c['confirmation_target_half_width'])**2)))
    summary=dict(stage=c['stage'],run=str(run.resolve()),families=len(c['task_seeds']),models=len(base),trajectories=len(records),
        prediction_records=len(records)*(c['update_steps']+1)*80,base=base_results,qualified_architectures=qualified_arch,
        primary=primary,descriptive=summaries,normalized_sensitivity=normalized,readout_controls=controls,matched_progress=progress,
        identity_max_error=identity,normalized_zero_gradient_steps=zero,proposed_confirmation_families=proposed,
        inference=dict(unit='task family; initialization seeds and update directions averaged within family',
            primary_interval_confidence=1-alpha,primary_familywise_alpha=.05,primary_interval_count=3*len(c['architectures']),
            primary_endpoints=['stable','changed','transfer'],secondary_interval_confidence=.95,
            exclusions='none; all planned models and both update directions retained'))
    s.write(output/'summary.json',summary);s.write(output/'records.json',records);s.write(output/'base-records.json',base)
    s.write(output/'curves.json',[dict(architecture=k[0],history=k[1],arm=k[2],metrics=['changed','stable','transfer','stable_drift'],
        mean=np.mean(v,axis=0).tolist()) for k,v in curves.items()])
    s.write(output/'provenance.json',dict(created_utc=s.now(),source_sha256=s.sha(__file__),run_manifest_sha256=s.sha(run/'manifest.json'),
        run_status_sha256=s.sha(run/'status.json'),summary_sha256=s.sha(output/'summary.json')))
    print(__import__('json').dumps({k:summary[k] for k in ['stage','families','models','base','qualified_architectures','primary','proposed_confirmation_families']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.run,a.output)
