# Changes from the original project

This release was exported from the project folder in which the studies ran. Files are copied byte for byte except
those listed below. There are two kinds of change:

- **redacted:** records in which personal or machine-specific strings were replaced: absolute paths and other
  projects' folder names by placeholders (`<project>`, `<portfolio>`, `<portfolio-root>`, `<workspace>`,
  `<research>`, `<environment-project>`, `<environment>`, `<sibling-project>`, `<uv-data>`, `<home>`), the operating
  system's name by `[distro]`, local timestamps by the same instant in UTC, and the names of the writing and coding tools in working records by the placeholder `[redacted]` (the assistants used are named in README.md, under "Use of AI assistants"). Nothing else in these files changed; the counts show how many strings of each kind were
  replaced.
- **not shipped:** files named in a protocol freeze but left out (listed at the end with their SHA-256).

Inputs larger than 1 MB, and the studies' evidence manifests, are shipped compressed (`.xz`); saved outputs are
archives under `data/saved-outputs/`. `tools/unpack.py` restores them byte for byte and checks each SHA-256 against
`data/PACKED-FILES.json`. Compression is not a change and is not listed below.

To check every protocol freeze against this release: `make check-freezes`.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `analysis/results/replay-2026-10-09.json` | redacted (absolute path x1, environment path x1) | `320d45f319e31c68141901bcb4b9b29fa730754b97c9aec62c709a8a5d6a1184` | - |
| `studies/1-feasibility-classification/02-Code/README.md` | redacted (environment path x2) | `749e740283330af511328bc8cf620d7d7ebf7e2976feb4ceacf8c2a572bdb06f` | - |
| `studies/1-feasibility-classification/02-Code/support.py` | adapted: the preparation manifest counts a file adapted for the release by its original's digest, as release-changes.json records it; the preparation writes the placeholder [time zone] where it wrote the local time zone, as the redacted records give it, so its --check still compares byte for byte | `b405f60393c511f6d096fbe0d7007a05e17b4ddee5f0980d33c5be58734900d4` | - |
| `studies/1-feasibility-classification/03-Data/prepared/preparation-manifest.json` | redacted (time zone x1) | `24e6c87eca3a4b487a93fcf3ecd23a68f0c7fbc7c5f6c0150c32888ae2230ce9` | - |
| `studies/1-feasibility-classification/04-Results/long-final-v1/manifest.json` | redacted (absolute path x1, environment path x1, platform string x1); shipped inside `data/saved-outputs/phase1-feasibility.tar.xz` | `097b5eb72433578a843cb82e74420173e7b6a8bcb0fc5aa7e2fa605f0a6aa25b` | - |
| `studies/1-feasibility-classification/04-Results/pilot-v1/manifest.json` | redacted (absolute path x1, environment path x1, platform string x1); shipped inside `data/saved-outputs/phase1-feasibility.tar.xz` | `0d78820bf98da5210b63b2ca2f83a23d1606efd72fc196a78cff34f1520fa341` | - |
| `studies/1-feasibility-classification/04-Results/qualification-v1/manifest.json` | redacted (absolute path x1, environment path x1, platform string x1); shipped inside `data/saved-outputs/phase1-feasibility.tar.xz` | `85dfe5f79ee7b791462ebd2c50cda3b6626629fd59a81b1296b7433eb067dfe0` | - |
| `studies/1-feasibility-classification/05-Notes/RUNBOOK.md` | redacted (machine path x1) | `0787cad825d749d218a225afcb391508830255a5b6b8f4c8c1a7feb349d8e6b9` | - |
| `studies/1-feasibility-classification/06-Study-History/ASSET-REFERENCES.json` | redacted (sibling project x2) | `6cb2cd66d45b178d6d5212714b6980c99052df078e57e4b0c664b96a73e1304c` | - |
| `studies/1-feasibility-classification/06-Study-History/IMPLEMENTATION-CHECKS.json` | redacted (environment path x1, project folder x1) | `53218c44a7cef5ac615070d30231ced01e6e4615d6a9450492cfb2af2f36ce54` | - |
| `studies/1-feasibility-classification/06-Study-History/PREPARATION-CHECK.json` | redacted (time zone x1) | `47fcc3b2487d8029646c0d78291dd365ef54b973ca05a0b99116e566ccef45ce` | - |
| `studies/1-feasibility-classification/06-Study-History/STATUS.json` | redacted (time zone x1) | `946f50d403da57d498cca3cd0a66a27e6eae9052f3fbfcccf469bdc87486ea45` | - |
| `studies/1-feasibility-classification/06-Study-History/environment.json` | redacted (absolute path x1, environment path x1) | `53fb585b479e3e49d7e821f65fe9dfc47a923ff4ba8fddf98ac671c171d1f9cc` | - |
| `studies/1-feasibility-classification/06-Study-History/pre-execution-guides-2026-10-08/05-Notes/RUNBOOK.md` | redacted (machine path x1) | `bbf1fe48abfebdf0de16ebeb8abcb484a5316952795a3d90c0d470692433cc48` | - |
| `studies/1-feasibility-classification/06-Study-History/pre-execution-guides-2026-10-08/06-Study-History/STATUS.json` | redacted (time zone x1) | `75f79aa6ec935163b8003deff4d249871f254bd830c1db9ca8575f07b9eb3f6b` | - |
| `studies/1-feasibility-classification/06-Study-History/pre-execution-guides-2026-10-08/README.md` | redacted (sibling project x1) | `477e5217e3a3273a6a3dae13f3d3b6e82bd9b8604621b9ac6be504a2f3ceefa1` | - |
| `studies/1-feasibility-classification/AMENDMENTS.md` | redacted (environment path x3, sibling project x2) | `ba0d77e95fd1af04ebd53b5771d40cd63f975540934e6ccd2fda11b7eeca903a` | - |
| `studies/1-feasibility-classification/Makefile` | redacted (environment path x1) | `145f5ac3ee694332d866c46a85eaabc03762cf4ec0bf18e7271f1fe4f59cfaca` | - |
| `studies/1-feasibility-classification/README.md` | redacted (sibling project x1) | `6e967adf212e22c52c37e6f01b3ed678b81389481f2a615952329f5f5b89a497` | - |
| `studies/1-feasibility-classification/REPRODUCIBILITY.md` | redacted (environment path x8) | `6858e1bdfa8d4c706d2b663e0aab76cf572a4c0ed0dd3f9e4a5ee042ec20f81a` | - |
| `studies/2-controlled-histories/AMENDMENTS.md` | redacted (home-relative path x1) | `097d9d38c19584cc5d8b6c3bc8d5b66f6a2aeac143098b0d4071bbe3afca3771` | - |
| `studies/2-controlled-histories/REPRODUCIBILITY-2026-10-09.md` | redacted (environment path x1) | `3a09bde81f8bf9080a42419a205c683fca6974a5c701bc9a28a06be262150c20` | - |
| `studies/2-controlled-histories/analysis/amendment/summary.json` | redacted (absolute path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `62d104b16ee0d82f53158b7fcb34bdb6e02e2f6507e771adeaeb4d6861b38355` | `studies/2-controlled-histories/protocol/CONFIRMATION-FREEZE.json` |
| `studies/2-controlled-histories/analysis/confirmation/summary.json` | redacted (absolute path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `7b4255cbe532727e166c5f7a40e1906d1b42325fb237e703e25bccd888e7fdd2` | `studies/2-controlled-histories/protocol/DURATION-FREEZE.json` |
| `studies/2-controlled-histories/analysis/development/summary.json` | redacted (absolute path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `f324bc9c50e833f7487b770bddf5bdb2eb83036806e86c395ac4bcff0928a907` | `studies/2-controlled-histories/protocol/CONFIRMATION-FREEZE.json` |
| `studies/2-controlled-histories/runs/amendment/manifest.json` | redacted (absolute path x1, environment path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `ff223d94d8f00c5e308a62889fa7fbba784450a0a5d9c59bde6eb589b221ed50` | - |
| `studies/2-controlled-histories/runs/confirmation/manifest.json` | redacted (absolute path x1, environment path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `6c752c2daf02a173b1a053c33dcaad8be1408d692fdf38b3131f0d48caf81ee5` | - |
| `studies/2-controlled-histories/runs/development/manifest.json` | redacted (absolute path x1, environment path x1); shipped inside `data/saved-outputs/phase2-controlled-histories.tar.xz` | `fbe1700169c6d7e9ad7379645ad09f8d37132c53b86abec2659ffe6dc85dff8e` | - |
| `studies/3-larger-confirmation/FREEZE.json` | redacted (platform string x1) | `45b029219b3510f527396a79048442f747473c08a726440012c5a1e8a4dce79f` | - |
| `studies/3-larger-confirmation/code/pipeline.sh` | adapted: the default interpreter is the release's environment (.venv at the release root) | `4fd1fa1cfa44a69e1a8eb36c5dca267404bdcb592c2b740a32674597658a5199` | `studies/3-larger-confirmation/FREEZE.json` |
| `studies/3-larger-confirmation/results/records/summary.json` | redacted (absolute path x1); shipped inside `data/saved-outputs/study3-larger-confirmation.tar.xz` | `06b573c54639f4532f097c0066258eb97999f19a43a7970da31d0c957b881d69` | - |
| `studies/3-larger-confirmation/runs/confirmation/manifest.json` | redacted (absolute path x1, environment path x1); shipped inside `data/saved-outputs/study3-larger-confirmation.tar.xz` | `b162554cb26bc05c9e463b1e79c0c008ce2722234d50e489651176c5cd916c97` | - |
| `studies/3-larger-confirmation/runs/confirmation/snapshot/FREEZE.json` | redacted (platform string x1); shipped inside `data/saved-outputs/study3-larger-confirmation.tar.xz` | `45b029219b3510f527396a79048442f747473c08a726440012c5a1e8a4dce79f` | - |
