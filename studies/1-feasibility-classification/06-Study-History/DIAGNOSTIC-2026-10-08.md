# AI-07 bounded diagnostic after the feasibility pilot

The complete pilot has been inspected. All 27 base networks achieve perfect accuracy on the two primitives and the 48 observed combinations. Mean accuracy on the 16 withheld combinations is 3.47% for factorwise, 5.56% for mixed-change and 2.78% for shuffled history. None of nine paired initializations qualifies. The primary mean factorwise advantage after updating is 2.78 percentage points, smaller than 10.31 points of within-family initialization variability. The frozen gate fails. Causal localization and independent confirmation are therefore not justified.

The independent audit checks 213,840 raw records, 405 final checkpoints and all 378 update/control trajectories. It reproduces the three qualification runs exactly. This diagnostic is selected after seeing the pilot and is development only.

## Longer final training

Repeat all 27 original instances with the same initial states, histories, learning rate and controls. Increase only the common final E00 block from 200 to 2,000 optimizer steps, giving 4,400 base steps per model. Keep the primary update budget at 100 steps. This tests whether more exposure to the final environment resolves the gross competence failure; it does not test arbitrarily long training, other optimizers or architectures. Keep the original results and stopping rule unchanged. If the competence gate still fails, stop the proposed MLP/history study.

## Explicit modular reference

Use two separate 8-by-8 learned linear lookup maps with a fixed modulo-eight composition rule. Fit the E00 primitive labels only for 200 Adam steps at learning rate 0.03. For each independent update, clone the base model, update only the relevant map using eight new primitive labels for 400 steps with fresh Adam, and keep the other map fixed. Evaluate both primitives and every combination at steps 0, 1, 5, 10, 25, 50, 100, 200 and 400. Use the original three task families and three initialization seeds, giving nine reference models and eighteen update trajectories. All combined labels are evaluation only.

The composition output is the circular convolution of the two predicted categorical distributions, computed in log space. The rule and separation are supplied by the architecture. The reference is an assay-sensitivity control with different capacity, learning rate and update budget, not evidence that training history discovered modules and not a fair architecture benchmark. A successful reference establishes that the supplied primitives contain enough information and the scoring can detect preserved, transferable updates. Its paired initializations do not create additional independent task families.

## Final decision

No further hyperparameter search or task redesign follows a second competence failure. Finish the research record with all outcomes, checks, figures, limitations and reproducibility instructions. The broad hypothesis can remain open even if this operationalization does not merit expansion.
