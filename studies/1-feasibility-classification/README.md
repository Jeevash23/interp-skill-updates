# AI-07 Independently changeable skills

**Feasibility study completed on 8 October 2026. The current MLP and learning-history design does not qualify for a mechanism study.** All 27 pilot networks learned the observed examples perfectly, but accuracy on unseen combinations averaged only 2.8–5.6% across histories. A longer final training period did not resolve this failure. A reference with explicit modular structure passed the same composition assay.

Read the [research report](01-Papers/AI-07-FEASIBILITY-REPORT.md) for the results and decision, or [reproducibility instructions](REPRODUCIBILITY.md) for the retained code, observations and checks. This is a negative feasibility result for this operationalization; it does not disprove the broader learning-history hypothesis.

| Record | Location |
|---|---|
| Results, interpretation and limits | [Feasibility report](01-Papers/AI-07-FEASIBILITY-REPORT.md) |
| Main figure | [Pilot and diagnostic results](04-Results/figures/feasibility.png) |
| Original research question | [Research brief](01-Papers/RESEARCH-BRIEF.md) |
| Rules fixed before model outcomes | [Execution protocol](06-Study-History/EXECUTION-2026-10-08.md) |
| Bounded diagnostic selected after the pilot | [Diagnostic protocol](06-Study-History/DIAGNOSTIC-2026-10-08.md) |
| Data and verification inventory | [Results guide](04-Results/README.md) |
| Code and reproduction | [Code guide](02-Code/README.md) · [Reproducibility](REPRODUCIBILITY.md) |
| Current machine-readable status | [Status](06-Study-History/STATUS.json) |

The completed work comprises 27 pilot runs, 27 longer-final-training runs and nine explicit modular reference models, plus three qualification runs reproduced exactly in the pilot. The pilot and longer-training diagnostic each contain 378 update/control trajectories. These reuse three task families; they are not independent confirmation samples. All runs used one CPU thread and no GPU.

Causal alignment, parameter restriction and independent confirmation were conditional on behavioral qualification and were not started. No scientific job or retry remains scheduled. The 7 October [draft](05-Notes/PROTOCOL-DRAFT.md), prepared inputs and preparation records remain unchanged as historical evidence.

[AI-08](../<sibling-project>/README.md) remains a separate prepared project. [Field guide](../README.md).
