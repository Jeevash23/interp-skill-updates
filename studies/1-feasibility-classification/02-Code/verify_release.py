"""Read-only verification of the completed AI-07 evidence inventory."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def verify():
    manifest=read(ROOT/'06-Study-History/FINAL-EVIDENCE-MANIFEST.json')
    for name,expected in manifest['files'].items():
        path=ROOT/name
        assert path.is_file(), f'Missing evidence: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, f'Changed evidence: {name}'
    for filename in ['EXECUTION-FREEZE-v1.json','ANALYSIS-FREEZE-v1.json','DIAGNOSTIC-FREEZE-v1.json']:
        for name,expected in read(ROOT/'06-Study-History'/filename)['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, f'Changed pre-outcome freeze: {name}'
    references=read(ROOT/'06-Study-History/ASSET-REFERENCES.json')['files']
    for reference in references:
        path=ROOT.parent/reference['path_from_field_root']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256'], f'Reference modified: {path}'
    audit_counts=[]
    for name,expected_jobs,expected_trajectories in [('qualification',3,42),('pilot',27,378),('long-final',27,378)]:
        state=read(ROOT/f'04-Results/{name}-v1/status.json')
        assert state['status']=='complete' and len(state['completed'])==expected_jobs and not state['failures']
        audit=read(ROOT/f'04-Results/{name}-audit-v1.json')
        assert audit['status']=='passed' and audit['model_jobs']==expected_jobs and audit['update_trajectories_checked']==expected_trajectories
        audit_counts.append(audit['raw_records_checked'])
    for name in ['pilot','long-final']:
        summary=read(ROOT/f'04-Results/{name}-analysis-v1/summary.json')
        assert summary['gate_passed'] is False and summary['decision']=='stop_mechanism_stage'
        assert summary['gate_criteria']['complete_coverage'] is True
    ref=read(ROOT/'04-Results/modular-reference-v1/summary.json')
    refaudit=read(ROOT/'04-Results/modular-reference-audit-v1.json')
    assert ref['status']=='complete' and len(ref['models'])==9 and ref['assay_passed']
    assert refaudit['status']=='passed' and refaudit['reference_assay_passed']
    audit_counts.append(refaudit['raw_records_checked'])
    assert sum(audit_counts)==465120
    state=read(ROOT/'06-Study-History/STATUS.json')
    assert state['stage']=='complete_negative_feasibility' and state['experiments_running']==0
    print(json.dumps(dict(status='passed',files_checked=len(manifest['files']),reference_files_unchanged=len(references),
                          audited_raw_records=sum(audit_counts),scientific_decision='stop_current_design'),indent=2))


if __name__=='__main__':verify()
