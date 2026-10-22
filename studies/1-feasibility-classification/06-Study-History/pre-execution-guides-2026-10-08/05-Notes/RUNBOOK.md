# Starting the independently changeable skills pilot

The machine was reserved for other work during setup. These instructions describe a later execution stage; nothing here starts automatically.

## Available now

`make check` validates the symbolic inputs, matching example counts, disjoint input splits, exact labels and reserved update functions. `make plan` prints planned model counts and schedules. Both use Python's standard library, without importing a model framework, querying the GPU or starting workers.

## Next work when resources are available

1. Review the unresolved transition-difficulty and optimizer-state choices in the [protocol](PROTOCOL-DRAFT.md). Complete the focused novelty check.
2. Implement a new local training runner and causal-analysis adapter according to [the execution contract](../02-Code/EXECUTION-CONTRACT.md). Reference AI-05 code read-only; do not rerun its original pipelines.
3. Record an isolated or explicitly reused environment without upgrading another project's environment. Choose a small development run and measure CPU/GPU time and memory. No performance estimate is currently verified.
4. Check task labels independently, no-update checkpoint identity, unchanged-function targets, and reproducibility. Use development-only runs to qualify competence and matching.
5. Resolve the draft, freeze code/configuration/input hashes and analysis rules, then run the bounded pilot in a fresh output directory. Preserve failures and interrupted states.
6. Write the feasibility decision before expanding seeds, architectures or task families.

## Resource coordination

No experiment is authorized to start during the current preparation request. After the user resumes experimental work, inspect current workload and use the existing portfolio GPU lock if a GPU is used. The known lock is `<coordination-folder>/2026-09-22/gpu.lock` relative to this project root. Acquire it nonblocking and defer if busy; do not replace it or create a competing lock. Never stop another job, install drivers or schedule a retry automatically. CPU training also consumes shared resources and is deferred.

For future execution, cap numerical thread pools and use one model worker. A failed memory check must stop before allocating model tensors. Keep checkpoints local to new run directories; there is no reason to download pretrained weights for the first MLP study.
