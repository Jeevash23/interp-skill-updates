# AI-07 execution protocol and authorization

The 8 October 2026 user request authorizes conducting AI-07, assessing whether it works and is worthwhile, and continuing to complete the research when justified. This supersedes the earlier preparation-only restriction. The scope is AI-07; the other portfolio projects remain reference material.

## Development pilot

Preserve the 7 October draft and prepared inputs. The executable settings are `03-Data/config/execution-v1.json`. Use the existing AI-05 CPU environment without installation or modification: Python 3.12.14, PyTorch 2.7.1+cpu and NumPy 2.2.0. One worker, one numerical thread, deterministic CPU float32, and reduced process priority limit interference with other work. Refuse to start with less than 2 GiB available memory or 2 GiB disk space. No GPU is needed.

Implement the draft 19-input, 64/64-hidden-unit, eight-output ReLU MLP. Train all 27 planned task/initialization/history instances for exactly 13 blocks of 200 full-batch Adam steps, learning rate 0.003. Do not select checkpoints by evaluation performance. The initial state is paired across histories. Log transition losses, target disagreement, gradient norms, per-block training behavior, optimizer state and final competence.

This is development, including the withheld argument pairs. The first task/initialization triplet may be used for runtime and numerical qualification; do not relabel it as independent confirmation. The main pilot repeats and retains all 27 instances. An exact replay of the qualification triplet measures implementation repeatability rather than adding independent samples.

## Update controls

From each common E00 checkpoint, make f2/g0 and f0/g2 branches. In each branch run 100 steps on eight changed-primitive examples only. Compare fresh Adam state (primary) with the inherited Adam state (diagnostic). For each optimizer state also run 100 steps on the corresponding old primitive and 100 steps on the complete old E00 training set. The latter matches optimizer steps, not examples per step; report this distinction. A no-update branch is recorded once per changed function with no optimizer, retaining the same parameters at every evaluation step.

This gives 108 changed-primitive trajectories, 108 old-primitive controls, 108 old-environment controls and 54 no-update controls: 378 trajectories overall. Evaluate at steps 0, 1, 5, 10, 25, 50 and 100. Score all trajectories against the branch's updated target and also report old-target accuracy to expose inertia. Store every example's logits and predictions. Training never consumes combined evaluation targets during a changed-primitive update.

## Qualification and decision rules

The following rules are fixed before viewing any model outcomes. They are practical feasibility thresholds, not a public preregistration or statistically powered population test.

A paired initialization qualifies only when every history has 100% accuracy on each primitive, at least 95% on observed combined pairs and at least 75% on withheld combined pairs before adaptation. Require at least two of three paired initializations in each task family. This competence requirement prevents interpreting preservation of an operation the model never learned compositionally as modular learning.

At step 100 under fresh Adam, factorwise history must attain at least 87.5% accuracy on the changed primitive, at least 87.5% on the unchanged primitive and at least 75% on updated withheld combinations, averaged within each task family over all three initializations and both update directions. These outcomes are reported separately. A proposed history advantage additionally needs at least a 10 percentage point improvement over shuffled history on updated withheld combinations in at least two families, no reversal larger than five points, and a mean advantage larger than within-family initialization variability. Report retained-optimizer sensitivity separately; disagreement requires further development rather than a mechanism claim. No-update and old-label controls must establish that any composition improvement is attributable to the changed labels.

The primary endpoint is the predeclared step 100; the other checkpoints describe trajectories. Show all seeds, all denominators and failures. The task family is the independent unit, with three families supporting descriptive comparisons only. No p-value or narrow bootstrap based on the number of examples or checkpoints is justified.

## Known design limitation

The symbolic audit, conducted before model execution, finds mean target-disagreement rates of 0.8203/0.7109/0.7109 for factorwise histories and 0.7188/0.5586/0.7174 for shuffled histories across the three tasks. Shuffled histories contain unchanged consecutive environments. Equal exposure does not equalize transition difficulty or recency. The pilot can establish feasibility and descriptive differences, but cannot isolate the cause as independent change. A successful pilot must be followed by a separately frozen transition-controlled design before causal attribution. Regression adjustment alone does not fix this issue.

If the behavioral gate fails, stop the strong mechanism investigation. One bounded, separately recorded development diagnostic may distinguish insufficient training or a broken assay from a failure of the proposed learning setup. A failed gate is a completed feasibility outcome, not a result to suppress or fix through open-ended parameter search. Positive results would require a targeted redesign, frozen causal interventions and fresh confirmation families; those stages are conditional.

## Numerical verification and provenance

Before scientific runs, check independent symbolic labels and input encoding, NumPy agreement with network forward passes, paired initial states, unchanged target identities, branch isolation, optimizer reset/retention, no-update identity, training-label isolation and deterministic repeats. Use explicit run commands and fresh directories, save source/configuration/input hashes and immutable snapshots, keep checkpoints and optimizer state, and retain interrupted or failed jobs. Independently recompute final statistics from per-example records and verify saved checkpoints before drawing conclusions.
