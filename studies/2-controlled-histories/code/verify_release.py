"""Freeze or verify the complete controlled-continuation evidence inventory."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
MANIFEST=ROOT/'FINAL-EVIDENCE-MANIFEST.json'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def checks():
    for filename in ['DEVELOPMENT-FREEZE.json','AMENDMENT-FREEZE.json','CONFIRMATION-FREEZE.json','VERIFICATION-FREEZE.json','DISPLAY-FREEZE.json','DURATION-FREEZE.json']:
        freeze=read(ROOT/'protocol'/filename)
        for rel,expected in freeze.get('files',freeze.get('source_files',{})).items():assert sha(ROOT/rel)==expected,(filename,rel)
    planned=dict(development=32,amendment=16,confirmation=512)
    counts=[];times=[]
    for stage,n in planned.items():
        state=read(ROOT/'runs'/stage/'status.json');assert state['status']=='complete' and not state['failures'] and len(state['completed'])==n
        summary=read(ROOT/'analysis'/stage/'summary.json');audit=read(ROOT/'analysis'/f'{stage}-audit.json')
        assert summary['models']==n and summary['trajectories']==18*n
        assert audit['status']=='passed' and audit['models']==n and not audit['failures']
        assert audit['per_example_trajectory_records']==n*18*201*80
        assert audit['checkpoints_verified']==n*48
        assert audit['audit_source_sha256']==sha(ROOT/'code/audit.py')
        counts.append(audit);times.append(state['seconds'])
        manifest=read(ROOT/'runs'/stage/'manifest.json')
        for rel,expected in manifest['source_hashes'].items():assert sha(ROOT/'runs'/stage/'snapshot'/rel)==expected
    freeze=read(ROOT/'protocol/CONFIRMATION-FREEZE.json');data=read(ROOT/'data/confirmation.json')
    assert freeze['created_utc']<data['created_utc']<read(ROOT/'runs/confirmation/manifest.json')['created_utc']
    assert not freeze['confirmation_outcomes_observed'] and not freeze['confirmation_tasks_generated']
    assert [t['seed'] for t in data['tasks']]==list(range(91001,91065))
    assert not set(t['seed'] for t in data['tasks'])&set(t['seed'] for t in read(ROOT/'data/development.json')['tasks'])
    replay=read(ROOT/'runs/selected-replay/verification.json');assert replay['status']=='passed' and replay['models']==8 and replay['bitwise_identical']
    analysis_replay=read(ROOT/'analysis/analysis-replay-verification.json');assert analysis_replay['status']=='passed'
    duration=read(ROOT/'runs/duration-control/summary.json');duration_audit=read(ROOT/'analysis/duration-audit.json')
    assert duration['status']=='complete' and duration['trajectories']==1024 and duration['independent_torch_replayed_trajectories']==16
    assert duration_audit['status']=='passed' and duration_audit['trajectories']==1024
    assert duration_audit['source_sha256']==sha(ROOT/'code/audit_duration.py')
    for rel,expected in duration['source_hashes'].items():assert sha(ROOT/rel)==expected
    for arch in ['linear','leaky']:
        for history in ['independent','joint']:
            row=duration['descriptive'][arch][history]
            assert row['endpoint']['stable_drift']['maximum']==0 and row['progress']['first_0_01']['reached']==256
    status=read(ROOT/'STATUS.json');assert status['experiments_running']==0 and status['study_execution_complete']
    assert (ROOT/'report/AI-07-COMPLETED-STUDY.md').is_file()
    old=subprocess.run([sys.executable,str(PROJECT/'02-Code/verify_release.py')],capture_output=True,text=True,check=True)
    return dict(status='passed',scientific_models=560,verification_retrained_models=8,trajectories=10080,
        audited_per_example_records=sum(a['per_example_trajectory_records'] for a in counts)+duration_audit['per_example_records'],
        audited_checkpoints=sum(a['checkpoints_verified'] for a in counts)+duration_audit['endpoint_checkpoints'],
        additional_duration_trajectories=1024,additional_duration_torch_replays=16,
        independent_numpy_trajectories_replayed=sum(a['independent_numpy_trajectories_replayed'] for a in counts),
        original_study_verification=json.loads(old.stdout),experimental_seconds=sum(times)+duration['seconds'])


def main(freeze):
    result=checks()
    if freeze:
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=MANIFEST and p.name!='RELEASE-VERIFICATION.json' and '__pycache__' not in p.parts}
        inventory=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,
            external_files={'../../CURRENT-STATUS.md':sha(PROJECT/'CURRENT-STATUS.md'),
                            '../../06-Study-History/FINAL-EVIDENCE-MANIFEST.json':sha(PROJECT/'06-Study-History/FINAL-EVIDENCE-MANIFEST.json')})
        with MANIFEST.open('x') as f:json.dump(inventory,f,indent=2,sort_keys=True);f.write('\n')
    inventory=read(MANIFEST)
    for rel,expected in inventory['files'].items():assert sha(ROOT/rel)==expected,rel
    for rel,expected in inventory['external_files'].items():assert sha(ROOT/rel)==expected,rel
    result.update(files_checked=len(inventory['files']),manifest_sha256=sha(MANIFEST))
    if freeze:(ROOT/'RELEASE-VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
