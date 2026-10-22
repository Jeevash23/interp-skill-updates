# AI-07 Prepared inputs

[config/pilot.json](config/pilot.json) preserves the original provisional design. [prepared/counts.json](prepared/counts.json) summarizes the preparation snapshot, including its historical zero-run status. Everything under `prepared/` is an input definition or plan, not an observed model outcome. Current execution status is in [Study History](../06-Study-History/STATUS.json).

[config/execution-v1.json](config/execution-v1.json) defines the completed development pilot and its stopping rules. [config/long-final-v1.json](config/long-final-v1.json) defines the separately recorded longer-final-training diagnostic. The original task families, input splits and histories were preserved in both. Model observations and checkpoints are under [Results](../04-Results/README.md).

The preparation manifest records exact code, configuration, protocol and output hashes. It is explicitly **not a scientific freeze or public preregistration**. Do not replace a prepared version after inspecting model outcomes; create a new version and record why it changed.

Only small JSON/JSONL files are stored here. Reference studies and their environments remain in their original locations. New checkpoints are local to AI-07's run directories. See [the code guide](../02-Code/README.md) for deterministic reconstruction.
