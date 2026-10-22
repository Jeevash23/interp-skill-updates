"""Publication-sized descriptive figures from saved AI-07 observations."""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

os.environ['MPLCONFIGDIR']='/tmp/ai07-matplotlib'
os.environ['OPENBLAS_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'04-Results'
DEST=RESULTS/'figures'
HISTORIES=['factorwise','mixed_change','shuffled']
LABELS={'factorwise':'Factorwise','mixed_change':'Mixed change','shuffled':'Shuffled'}
COLORS={'factorwise':'#087f8c','mixed_change':'#ad6b1a','shuffled':'#7553a3'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})


def csvrows(path):
    with path.open() as f:return list(csv.DictReader(f))


def mean(rows,key):return statistics.mean(float(r[key]) for r in rows)


def make_figures():
    DEST.mkdir(exist_ok=True)
    base=csvrows(RESULTS/'pilot-analysis-v1/base.csv')
    extended=csvrows(RESULTS/'long-final-analysis-v1/base.csv')
    endpoints=csvrows(RESULTS/'pilot-analysis-v1/endpoints.csv')
    curves=csvrows(RESULTS/'pilot-analysis-v1/curves.csv')
    effects=csvrows(RESULTS/'pilot-analysis-v1/paired-effects.csv')
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.8),layout='constrained')
    ax=axes[0,0]
    for j,(data,key,label,color) in enumerate([(base,'combined_seen','Observed combinations','#556875'),
                                             (base,'combined_heldout','Unseen combinations','#087f8c'),
                                             (extended,'combined_heldout','Unseen, longer final training','#87c4c7')]):
        values=[mean([r for r in data if r['history']==h],key)*100 for h in HISTORIES]
        x=np.arange(3)+(j-1)*.24
        ax.bar(x,values,.22,color=color,label=label)
        for xx,yy in zip(x,values):ax.text(xx,yy+2,f'{yy:.1f}',ha='center',fontsize=8)
    ax.axhline(75,color='#888888',ls='--',lw=1,label='75% competence threshold')
    ax.set(xticks=np.arange(3),xticklabels=[LABELS[h] for h in HISTORIES],ylim=(0,117),ylabel='Accuracy (%)',title='A  Memorization without composition')
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.12),frameon=False,ncol=1)

    ax=axes[0,1]
    metrics=['changed_primitive','unchanged_primitive','combined_heldout']
    for j,h in enumerate(HISTORIES):
        rows=[r for r in endpoints if r['history']==h and r['condition']=='changed_primitive' and r['optimizer_mode']=='reset']
        values=[mean(rows,k)*100 for k in metrics]
        x=np.arange(3)+(j-1)*.24
        ax.bar(x,values,.22,color=COLORS[h],label=LABELS[h])
        for xx,yy in zip(x,values):ax.text(xx,yy+2,f'{yy:.1f}',ha='center',fontsize=8)
    ax.set(xticks=np.arange(3),xticklabels=['Updated skill','Preserved skill','Updated unseen\ncombinations'],ylim=(0,117),ylabel='Accuracy (%)',title='B  Primitive learning does not transfer')
    ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(0,-.12),frameon=False,ncol=3)

    ax=axes[1,0]
    steps=[0,1,5,10,25,50,100]
    for h in HISTORIES:
        selected=[r for r in curves if r['history']==h and r['condition']=='changed_primitive' and r['optimizer_mode']=='reset']
        y=[mean([r for r in selected if int(r['step'])==s],'combined_heldout')*100 for s in steps]
        ax.plot(steps,y,marker='o',ms=4,lw=2,color=COLORS[h],label=LABELS[h])
    ax.axhline(12.5,color='#999999',ls=':',lw=1,label='Uniform guessing reference')
    ax.set(xlabel='Update steps',ylabel='Updated unseen-combination accuracy (%)',ylim=(0,40),title='C  Low accuracy throughout adaptation')
    ax.legend(fontsize=8,frameon=False,loc='upper right')

    ax=axes[1,1]
    tasks=sorted({int(r['task_seed']) for r in effects})
    for j,mode in enumerate(['reset','retained']):
        color='#087f8c' if mode=='reset' else '#ca6456'
        means=[]
        for i,task in enumerate(tasks):
            values=[float(r['factorwise_minus_shuffled'])*100 for r in effects if int(r['task_seed'])==task and r['optimizer_mode']==mode]
            x=i+(j-.5)*.25
            ax.scatter([x-.045,x,x+.045],values,s=25,alpha=.6,color=color)
            means.append(statistics.mean(values))
        ax.plot(np.arange(3)+(j-.5)*.25,means,marker='D',color=color,lw=1.5,label=f'{mode.capitalize()} Adam state')
    ax.axhline(0,color='#777777',lw=1)
    ax.axhline(10,color='#999999',ls='--',lw=1,label='10-point effect threshold')
    ax.set(xticks=np.arange(3),xticklabels=['Task 1','Task 2','Task 3'],ylabel='Factorwise minus shuffled (percentage points)',title='D  Small, optimizer-sensitive differences',ylim=(-24,24))
    ax.legend(fontsize=8,frameon=False,loc='lower left')
    fig.suptitle('AI-07 feasibility result: the proposed MLP does not learn reusable composition',fontsize=15,fontweight='bold')
    fig.savefig(DEST/'feasibility.png',dpi=180)
    fig.savefig(DEST/'feasibility.svg')
    fig.savefig(DEST/'feasibility.pdf')
    plt.close(fig)

    reference=json.loads((RESULTS/'modular-reference-v1/summary.json').read_text())
    fig,ax=plt.subplots(figsize=(8.5,4.8),layout='constrained')
    for mode,ls in [('reset','-'),('retained','--')]:
        for h in HISTORIES:
            rows=[r for r in curves if r['history']==h and r['optimizer_mode']==mode and r['condition']=='changed_primitive']
            values=[mean([r for r in rows if int(r['step'])==s],'unchanged_primitive')*100 for s in steps]
            ax.plot(steps,values,color=COLORS[h],ls=ls,lw=2,label=f'{LABELS[h]}, {mode}')
    ax.set(xlabel='Update steps',ylabel='Unchanged primitive accuracy (%)',ylim=(0,105),title='Retaining Adam state increases collateral forgetting in this pilot')
    ax.legend(fontsize=8,frameon=False,ncol=2,loc='lower left')
    fig.savefig(DEST/'optimizer-controls.png',dpi=180)
    fig.savefig(DEST/'optimizer-controls.svg')
    plt.close(fig)
    print(json.dumps({'figures':['feasibility.png','feasibility.svg','feasibility.pdf','optimizer-controls.png','optimizer-controls.svg'],
                      'source':'saved development results','independent_task_families':3,'reference_assay_passed':reference['assay_passed']}))


if __name__=='__main__':make_figures()
