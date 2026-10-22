# AI-07 Independently changeable skills

Can learning history make one learned operation easier to change without disrupting another? This pilot separates **modularity of computation** from **modularity of learning**.

**Status on 7 October 2026: preparation only.** The draft design, symbolic task generator, configurations and checks are available. No network has been trained, no model has been loaded and no scientific results exist. Training and causal-intervention runners still require implementation and qualification before experiments. The present setup does not schedule or start them.

| Read | Location |
|---|---|
| Research question and contribution | [Research brief](01-Papers/RESEARCH-BRIEF.md) |
| Draft pilot design and decision rules | [Protocol](05-Notes/PROTOCOL-DRAFT.md) |
| Closest work and novelty boundaries | [Literature](05-Notes/LITERATURE.md) |
| What to do when compute is available | [Runbook](05-Notes/RUNBOOK.md) |
| Lightweight preparation commands | [Code guide](02-Code/README.md) |
| Planned inputs | [Data guide](03-Data/README.md) |
| Evidence status | [Results](04-Results/README.md) |
| Setup record | [History](06-Study-History/README.md) |

Run `make check` here for a standard-library-only check of the prepared task definitions and plans. The default `make` only prints help. There is no training command in this scaffold.

AI-08 asks how context controls the use of an existing representation. The projects can share ideas and task definitions later, but neither depends on the other producing a positive result. [AI-08](../<sibling-project>/README.md) · [Field guide](../README.md).
