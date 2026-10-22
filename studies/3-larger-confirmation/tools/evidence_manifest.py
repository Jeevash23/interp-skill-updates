"""Write or check EVIDENCE-MANIFEST.json: the SHA-256 of every file of Study 3, written after the verification.

Added after the run (it is not part of the frozen code in code/). The first manifest, of 10:34 UTC on 10 October 2026,
is kept in provenance/; see AMENDMENTS.md.

    python3 -B tools/evidence_manifest.py            # write (refuses if the manifest exists)
    python3 -B tools/evidence_manifest.py --check    # every listed file present with its hash; no file unlisted
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
MANIFEST = STUDY / "EVIDENCE-MANIFEST.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory() -> dict:
    return {str(p.relative_to(STUDY)): sha(p) for p in sorted(STUDY.rglob("*"))
            if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}


def main() -> None:
    if "--check" in sys.argv:
        recorded = json.loads(MANIFEST.read_text())["files"]
        current = inventory()
        bad = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        print(f"{'ok  ' if not bad else 'FAIL'} study 3 evidence manifest: {len(recorded) - len(bad)}/{len(recorded)} files match"
              + (f"; differing: {bad[:10]}" if bad else ""))
        sys.exit(1 if bad else 0)
    if MANIFEST.exists():
        sys.exit("EVIDENCE-MANIFEST.json exists")
    files = inventory()
    MANIFEST.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), files=files), indent=2) + "\n")
    print(f"wrote {MANIFEST.name}: {len(files)} files")


if __name__ == "__main__":
    main()
