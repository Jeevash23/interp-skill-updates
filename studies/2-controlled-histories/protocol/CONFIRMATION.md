# Fresh confirmation, frozen before its tasks or outcomes

## Selection and fixed budget

The initial development cohort contained four task families, two initializations, two histories and two architectures (32 models). Deep linear passed all 8/8 paired qualification checks; standard ReLU passed 0/8. The one permitted diagnostic changed only the activation to LeakyReLU with slope 0.9 (16 models), passing 8/8 pairs. Select deep linear and this explicitly near-linear model. Retain and report the ReLU failure; there is no further architecture search.

Development family-level standard deviations of independent-minus-joint unchanged-skill RMSE were 0.3533609937654171 (linear) and 0.17719100232619497 (near-linear). The original rule ceil((2.3*max_SD/0.025)^2), bounded to 32–64, selects **64** independent fresh families, seeds **91001–91064**. Use initialization seeds 2701 and 2702, both histories, both architectures: **512 models and 9,216 adaptation trajectories**, each with 201 recorded states. Do not extend based on confirmation results. With the development variance, the cap may leave intervals wider than the desired 0.025 half-width. Precision must be reported honestly.

The 64 task tables are generated only after this document, configuration, experiment implementation and analysis are hashed. All original training, withholding, history, final-suffix and update settings remain fixed. Development outcomes cannot enter the confirmation estimates.

## Primary inference

The experimental unit is a task family. Within a family, average the two initialization seeds and both changed-function directions. Subtract joint-history error from independent-history error. A negative preservation difference favors independent changes. Retain every planned model and both directions, including any poor training or adaptation result. No outcome-based exclusion, checkpoint selection or optional stopping is allowed. Run failures prevent claiming the planned confirmation completed until resolved and disclosed.

Three endpoints are evaluated at the fixed step 200 under unrestricted SGD: unchanged-primitive RMSE; changed-primitive RMSE; updated unseen-combination RMSE divided by sqrt(2). Primitive normalization is one, the RMS of each target function. The combined normalizer is a fixed reference scale sqrt(2), not each held-out subset's empirical RMS.

Use paired family means and two-sided Student t intervals. Bonferroni correction covers all three primary endpoints across both selected architectures: six intervals, each at **99.1666667%**, giving at least 95% simultaneous nominal coverage under the t-model assumptions. Report raw family differences and their standard deviations. The small model-seed set limits inference over initialization; the 64 families, not 256 models per architecture or thousands of trajectories, supply independent replication. Report distribution plots and a bootstrap sensitivity analysis as descriptive, without changing the decision rule.

A practically useful independent-history result requires all of: unchanged-skill difference upper bound below -0.05; changed-skill difference upper bound below +0.02; unseen-combination difference upper bound below +0.02; and base qualification in at least 90% of paired confirmation instances. These are effect-size and noninferiority criteria, not a search for p<0.05. A preservation interval entirely inside [-0.05,+0.05] supports equivalence at that resolution. An interval wholly above +0.05 supports practical harm. If meaningful benefit is excluded but equivalence is not established, say so. All other results are inconclusive. Failure of base competence invalidates a selective-compositional-plasticity conclusion for that architecture, while all outcomes remain reported.

## Mechanism and sensitivity

Report all nine update arms. The normalized-gradient arm fixes path length to 2.0 over 200 steps unless gradients are zero; disclose exceptions. The first step at changed-skill RMSE <=0.10 is a progress-matched diagnostic; report success counts alongside conditional errors, without silently removing failures. No-update and old-label controls are scored against the new task and cannot count as successful adaptation merely because they preserve the unchanged skill.

Check the structural identity: unchanged-primitive output drift equals its fixed hidden-activation matrix times the readout-weight change. Check that the unchanged input columns never move, and that freezing the readout gives zero drift. Compare freezing the 32 readout weights against three random masks freezing 32 potentially active weights. Random masks match count, not functional rank. Readout-only updates test what that channel alone can achieve. Report adaptation and combined transfer under interventions to distinguish preservation from inability to learn.

For these prespecified secondary/control outcomes show descriptive 95% family intervals; they are not additional multiplicity-adjusted primary tests. The structural identity is an algebraic property of this model and optimizer, not an empirical discovery of a universal biological mechanism. Interpretation of the history effect remains separate from confirming this identity.

## Verification and closure

Independently reconstruct all task labels, target matching, base and per-environment checkpoint predictions, adaptation endpoint predictions, summary errors and family-level estimates from saved arrays. Check paired initial weights, all no-update trajectories, fixed-readout trajectories, random-mask frozen weights, unchanged input columns, step lengths, progress thresholds and the drift identity. Replay selected fresh models and adaptation trajectories deterministically from the saved source/configuration. Report maximum numerical discrepancy and failed checks.

Deliver a combined report covering the original classification failure, this development/amendment, fresh confirmation, causal controls, uncertainty, artifacts and reproduction instructions. Publish a frozen evidence manifest and preserve the original study's manifest unchanged. The bounded study can finish with an inconclusive history effect; that is not grounds to claim the unrestricted research question has been settled.
