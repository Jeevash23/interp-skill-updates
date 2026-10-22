# AI-07 Prepared inputs

[config/pilot.json](config/pilot.json) contains the provisional design. [prepared/counts.json](prepared/counts.json) summarizes generated inputs and planned work. Everything under `prepared/` is an input definition or plan, not an observed model outcome.

The preparation manifest records exact code, configuration, protocol and output hashes. It is explicitly **not a scientific freeze or public preregistration**. Do not replace a prepared version after inspecting model outcomes; create a new version and record why it changed.

Only small JSON/JSONL files are stored here. Existing checkpoints and frozen studies remain in their original locations. No weights or environments are copied. See [the code guide](../02-Code/README.md) for deterministic reconstruction.
