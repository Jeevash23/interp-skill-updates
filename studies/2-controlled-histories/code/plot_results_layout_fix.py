# Layout-only derivative of the preserved display-freeze source.
"""Scientific figures and prespecified descriptive sensitivities from frozen results."""
from collections import defaultdict
import json
import os
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())


def make():
    s=read(ROOT/'analysis/confirmation/summary.json');records=read(ROOT/'analysis/confirmation/records.json')
    curves=read(ROOT/'analysis/confirmation/curves.json');out=ROOT/'report/figures';out.mkdir(exist_ok=True)
    arches=['linear','leaky'];names={'linear':'Deep linear','leaky':'Mildly nonlinear'};colors={'linear':'#176a9d','leaky':'#cb6f20'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    arms=['unrestricted','freeze_readout','random_mean','readout_only','normalized','no_update','old_primitive']
    labels=['Unrestricted','Freeze readout','Random freeze (3 masks)','Readout only','Fixed-length steps','No update','Old labels']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,metric,title in zip(axes.flat,['stable','changed','transfer'],['A  Unchanged skill: error after updating','B  Changed skill: error after updating','C  Unseen combinations: updated-target error']):
        for ai,arch in enumerate(arches):
            means=[];halves=[]
            for arm in arms:
                grouped=defaultdict(list)
                for r in records:
                    chosen=r['arm'].startswith('random_freeze_') if arm=='random_mean' else r['arm']==arm
                    if r['architecture']==arch and chosen:grouped[r['task_seed']].append(r[metric])
                family=np.array([np.mean(v) for v in grouped.values()]);means.append(family.mean())
                halves.append(t.ppf(.975,len(family)-1)*family.std(ddof=1)/np.sqrt(len(family)))
            y=np.arange(len(arms))+(ai-.5)*.3
            ax.errorbar(means,y,xerr=halves,fmt='o',markersize=5,capsize=2,color=colors[arch],label=names[arch])
        ax.set_yticks(range(len(arms)),labels);ax.invert_yaxis();ax.set_xlabel('Normalized RMSE (lower is better)')
        ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='x',alpha=.16)
    ax=axes[1,1];rng=np.random.default_rng(62001)
    ax.axvspan(-.05,.05,color='#e8ecf0',label='±0.05 practical margin');ax.axvline(0,color='#777',lw=.8)
    for ai,arch in enumerate(arches):
        e=s['primary'][arch]['effects']['stable'];vals=list(e['family_differences'].values())
        ax.scatter(vals,ai+rng.uniform(-.13,.13,len(vals)),s=15,alpha=.45,color=colors[arch])
        ax.errorbar(e['mean'],ai,xerr=[[e['mean']-e['low']],[e['high']-e['mean']]],fmt='D',color='black',capsize=5,zorder=5)
    ax.set_yticks([0,1],[names[a] for a in arches]);ax.set_ylim(-.6,1.6)
    ax.set_xlabel('Independent history − joint history\nUnchanged-skill error; negative favors independent')
    ax.set_title('D  History effects across 64 fresh families',loc='left',fontsize=11);ax.grid(axis='x',alpha=.16)
    axes[0,0].legend(frameon=False,loc='lower right')
    fig.suptitle('AI-07: controlled confirmation and update interventions',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.045,1,.92))
    fig.text(.5,.015,'A–C: histories pooled within each family; descriptive 95% intervals. D: each dot is one family; black intervals are 99.17%.',ha='center',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'confirmation.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    hcolors={'independent':'#176a9d','joint':'#cb6f20'}
    for row,arch in enumerate(arches):
        for col,metric in enumerate(['changed','stable','transfer']):
            ax=axes[row,col]
            for history in ['independent','joint']:
                for arm,style in [('unrestricted','-'),('freeze_readout','--')]:
                    c=next(c for c in curves if c['architecture']==arch and c['history']==history and c['arm']==arm)
                    vals=c['mean'][c['metrics'].index(metric)]
                    ax.plot(range(len(vals)),vals,color=hcolors[history],linestyle=style,lw=1.5,label=f"{history.title()}, {'free readout' if arm=='unrestricted' else 'frozen readout'}")
            ax.set_title(f"{names[arch]} · {['Changed skill','Unchanged skill','Unseen combinations'][col]}",fontsize=10)
            ax.set_xlabel('Update steps');ax.set_ylabel('Normalized RMSE');ax.grid(alpha=.15)
    handles,leglabels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,leglabels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Learning the new skill and preserving the old one',fontsize=15)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'trajectories.{ext}')
    plt.close(fig)
    # Descriptive bootstrap; no alteration to the frozen t-interval decisions.
    rng=np.random.default_rng(61123);indices=rng.integers(0,s['families'],size=(50000,s['families']));sensitivity={}
    for arch in arches:
        sensitivity[arch]={}
        for metric in ['stable','changed','transfer']:
            d=np.array(list(s['primary'][arch]['effects'][metric]['family_differences'].values()))
            boot=d[indices].mean(1);alpha=.05/6
            sensitivity[arch][metric]=dict(percentile_interval_99_1667=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
                family_min=float(d.min()),family_max=float(d.max()),families_favoring_independent=int((d<0).sum()),families=len(d))
        for seed in [2701,2702]:
            family=defaultdict(lambda:defaultdict(list))
            for r in records:
                if r['architecture']==arch and r['model_seed']==seed and r['arm']=='unrestricted':family[r['task_seed']][r['history']].append(r['stable'])
            d=np.array([np.mean(v['independent'])-np.mean(v['joint']) for v in family.values()]);half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            sensitivity[arch][f'initialization_{seed}']=dict(mean=float(d.mean()),descriptive_95_interval=[float(d.mean()-half),float(d.mean()+half)])
        for branch in ['f','g']:
            family=defaultdict(lambda:defaultdict(list))
            for r in records:
                if r['architecture']==arch and r['branch']==branch and r['arm']=='unrestricted':family[r['task_seed']][r['history']].append(r['stable'])
            d=np.array([np.mean(v['independent'])-np.mean(v['joint']) for v in family.values()])
            sensitivity[arch][f'direction_{branch}']=dict(mean=float(d.mean()),families=len(d))
    (ROOT/'analysis/confirmation/descriptive-sensitivity.json').write_text(json.dumps(dict(bootstrap_seed=61123,resamples=50000,
        confirmatory=False,note='Bootstrap and subgroup diagnostics do not replace the prespecified primary intervals.',results=sensitivity),indent=2)+'\n')
    print('Saved six figure files and descriptive sensitivity results.')


if __name__=='__main__':make()
