# Execution adapter contract

Only preparation code is implemented. The model runner is a later development task; this contract prevents the draft plan from being confused with a validated experiment.

## Inputs and training

Read the prepared task family, its history order, initialization seed and configuration hash. Model inputs contain only query kind and arguments, never environment ID, function tables or target values. Build a training batch from the current environment's `train` examples. Repeated visits reuse exactly the same examples. All three histories for a task/seed start from identical parameters.

Use a bounded fixed step count, explicit dtype/device, controlled thread pools and a unique run directory created exclusively. Persist model and optimizer states, per-environment losses, transition disagreement, observed loss jumps, gradient norms, timing and failures. Do not overwrite a completed run or treat an incomplete checkpoint as final.

## Update branches

Clone the final common-environment checkpoint for every branch. Expose only the changed primitive's labels to the optimizer. Log accuracy on changed/unchanged primitives and combined targets at every declared checkpoint. Evaluation data never enter the training loss. Compute all retained target definitions from the corresponding environment, rather than reusing pre-update labels for updated compositions.

## Output record

Each record needs task seed, model seed, history, environment, branch, step, split, query kind, example ID, predicted class, target class, logits, status, input hashes and elapsed time. Preserve per-example records as well as summaries. The training run is the unit containing repeated checkpoints; inputs/checkpoints are not independent model replicates.

The causal adapter additionally records alignment fit split, frozen projection or neuron indices, source/recipient IDs, complete intervention parameters and null controls. Alignment fitting must not use final update evaluation outcomes. Numerical qualification of every intervention path is required before scientific interpretation.
