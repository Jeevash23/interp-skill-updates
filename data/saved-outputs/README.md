# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `phase1-feasibility.tar.xz` | 231 | 0.4 | 12.3 | Phase 1: run manifests, training and update records, analyses, audits and verification (network checkpoints and compressed training traces not included) |
| `phase2-controlled-histories.tar.xz` | 3791 | 40.0 | 72.1 | Phase 2: every run's JSON records and code snapshot, the confirmation networks' initial and base parameters, the 1,000-step frozen-readout runs, analyses, audits and verification (other network arrays not included) |
| `study3-larger-confirmation.tar.xz` | 3689 | 4.9 | 54.2 | Study 3: every run's JSON records, the extracted records, the analysis and the independent verification (network arrays not included) |

Total packed size: 45.3 MB. Rendered figures are not included; `make figures` draws them from these outputs.
