"""Small deterministic preparation helpers, using only the Python standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "03-Data/config/pilot.json"
DEFAULT_OUTPUT = ROOT / "03-Data/prepared"


def encode(name, value):
    if name.endswith(".jsonl"):
        return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in value)
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def release_digest(path):
    """release: a file adapted for the release counts by its original's digest, which release-changes.json
    records, so the recorded manifest still matches."""
    import sys
    sys.path.insert(0, str(ROOT.parents[1] / "tools"))
    import release_paths
    data = Path(path).read_bytes()
    change = release_paths.changes().get("changed_files", {}).get(release_paths.release_relative(path))
    if change and change["release_sha256"] == digest_bytes(data):
        return change["original_sha256"]
    return digest_bytes(data)


def preparation_main(build):
    parser = argparse.ArgumentParser(description="Prepare or check symbolic inputs; no model execution")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write to a new output directory; refuse overwrites")
    mode.add_argument("--check", action="store_true", help="Verify recorded inputs and deterministic outputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    bundle = build(config)
    content = {name: encode(name, value).encode() for name, value in bundle.items()}
    inputs = [CONFIG, ROOT / "02-Code/prepare.py", Path(__file__).resolve(),
              ROOT / "05-Notes/PROTOCOL-DRAFT.md"]
    manifest = dict(schema_version=1, project_id=config["project_id"],
                    record_type="preparation_snapshot_not_scientific_freeze",
                    created_date="2026-10-07", date_timezone="[time zone]",  # release: the local time zone is not given
                    model_execution_performed=False, tokenizer_loaded=False,
                    inputs={str(p.relative_to(ROOT)): release_digest(p) for p in inputs},
                    outputs={name: digest_bytes(data) for name, data in content.items()})
    content["preparation-manifest.json"] = encode("manifest.json", manifest).encode()
    if args.write:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, data in content.items():
            with (args.output / name).open("xb") as handle:
                handle.write(data)
    if args.check:
        if not args.output.is_dir():
            raise SystemExit("Prepared data missing. Use --write with a new output directory.")
        expected = set(content)
        actual = {p.name for p in args.output.iterdir()}
        if actual != expected:
            raise SystemExit(f"Prepared file inventory changed: {actual ^ expected}")
        for name, expected_data in content.items():
            if (args.output / name).read_bytes() != expected_data:
                raise SystemExit(f"Prepared input/output changed: {name}; review before writing a new version")
    print(json.dumps(dict(bundle["counts.json"],
        preparation_check="passed" if args.check else "not_requested",
        files_written=len(content) if args.write else 0), indent=2))
