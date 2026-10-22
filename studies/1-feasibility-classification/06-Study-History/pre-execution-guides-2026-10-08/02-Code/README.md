# AI-07 Preparation code

The scripts require Python 3.11 or newer and only its standard library. No installation is needed. They do not import PyTorch, NumPy, Transformers or tokenizer libraries, load weights, create workers or make network requests.

From this project root:

```sh
make plan
make check
```

Prepared inputs are already supplied. To regenerate into a **new** directory without replacing them:

```sh
python3 -B 02-Code/prepare.py --write --output /tmp/ai-07-new-preparation
python3 -B 02-Code/prepare.py --check --output /tmp/ai-07-new-preparation
```

`prepare.py` constructs symbolic inputs and a future job/condition plan. `support.py` records their checksums and refuses overwrite. `verify.py` independently checks the saved default inputs against task semantics and grouping constraints. `make check` performs both deterministic reconstruction and semantic verification. These checks do not qualify model behavior or numerical interventions.

There is no model runner in this setup. Its required behavior is specified in [EXECUTION-CONTRACT.md](EXECUTION-CONTRACT.md). The [runbook](../05-Notes/RUNBOOK.md) lists the remaining implementation and qualification steps. No hidden automatic execution or scheduler is included.
