# Results

Short account of what AI-07 found. The manuscript (`paper/manuscript.pdf`) has the details; every number below is
taken from the macros that `paper/make_numbers.py` generates from the saved outputs. Errors are root-mean-square
errors (RMSE) on functions of unit scale; differences are independent minus joint history, so a negative value
favours the independent history. Intervals for the primary comparison are simultaneous 95% intervals (each at
99.17%, six intervals).

## The question

If two skills often change separately during training, does the network become able to update one of them while
leaving the other intact? A network answers f(a), g(b) and f(a) + g(b) for two functions on eight symbols, with 16
of the 64 pairs withheld from training. Two histories visit the same 29 environments and differ in how often a single
function changes (22 of 28 transitions in the independent history, 4 in the joint one). After training, one function
is replaced by a new one and the network is updated by plain gradient descent on its eight new values for 200 steps.

## Phase 1: a classification version that failed to compose (manuscript Appendix A)

- 27 ReLU networks (three task families, three seeds, three histories) answered every training example correctly
  but scored 2.8–5.6% on withheld pairs, below the 12.5% of guessing; the best network reached 25%.
- Training the last environment ten times longer left withheld accuracy at 4.2–5.6%.
- A reference model with the modular structure built in answered every query and kept the untouched table bit for
  bit, so the assay detects composition when the structure is supplied.
- The phase stopped at its competence requirement before any history comparison.

## Phase 2: controlled histories

**Competence.** All 512 confirmation networks (64 fresh families, two seeds, two histories, linear and leaky ReLU of
slope 0.9) met the prespecified thresholds. The linear networks fit every query, including withheld pairs, to within
2 × 10⁻¹⁶; the leaky ones erred on withheld pairs by 0.0102 on average and 0.0207 at most.

**No history effect (primary result).**

| | Independent | Joint | Difference | Prespecified interval | Two intervals (post hoc) |
|---|---|---|---|---|---|
| Linear, damage | 0.671 | 0.674 | −0.003 | [−0.062, +0.056] | [−0.053, +0.047] |
| Leaky ReLU, damage | 0.675 | 0.668 | +0.008 | [−0.049, +0.065] | [−0.040, +0.056] |

- Both histories learn the changed function to about 0.0005; its difference interval lies within ±0.00013.
- Linear: inconclusive at the ±0.05 margin. Leaky: a benefit of 0.05 or more is excluded (lower bound −0.049, 0.0007 above −0.05). Neither interval lies
  inside ±0.05, so equivalence is not established. 30 (linear) and 23 (leaky) of the 64 families favoured the
  independent history.
- The development estimates (−0.091 linear, −0.143 leaky, four families) did not replicate.
- Post hoc: the error on withheld pairs equals the damage divided by √2 (correlation 0.9999999 linear, 0.9995 leaky),
  and the changed-skill error sits at its floor, so two of the three primary endpoints add no evidence. Correcting
  for the damage endpoint alone gives the intervals in the last column; the conclusions are the same.

**Interference passes through the readout.** An update on f queries leaves the input weights of g fixed, so the
change in every g answer equals g's hidden activations times the change in the readout; the saved updates obey this
to within 1.8 × 10⁻¹⁵.

- Freezing the 32 readout weights leaves the unchanged answers exactly as they were.
- Freezing 32 random weights among the 288 an update can move leaves the damage at 0.68 (unrestricted: 0.67).
- Updating the readout alone raises the damage to 1.16 and learns the new function poorly (0.26–0.28).

**Cost of freezing the readout.** At 200 steps, frozen-readout updates reach a changed-skill error of 0.199 (linear)
and 0.215 (leaky), and 0 of 1,024 reach 0.10; unrestricted updates reach 0.10 after 68–73 steps, by which point the
damage is already 0.65–0.66. In 1,000-step frozen-readout updates (declared after the confirmation results) all
1,024 reach 0.10 by step 343 and 0.01 by step 640, with the unchanged answers exact throughout: 3.9–4.0 times as many
steps as an unrestricted update. That they succeed follows from the architecture; the slowdown is the measurement.
Post hoc, the readout norm predicts each linear update's first step at 0.10 and 0.01 exactly (512 of 512).

**Secondary results.**

- Fixed-length steps (prespecified, descriptive 95% intervals): for the same path length in weight space, the
  independent history ended with a lower changed-skill error, −0.027 [−0.040, −0.015] (linear) and
  −0.026 [−0.040, −0.012] (leaky); damage differences +0.006 [−0.034, +0.047] and +0.023 [−0.018, +0.063].
- Update direction (post hoc, 95%): the independent history changes f alone 14 times and g alone 8 times. For
  f-updates the difference in damage was −0.038 [−0.103, +0.028] (linear) and −0.023 [−0.087, +0.041] (leaky); for
  g-updates +0.031 [−0.036, +0.098] and +0.038 [−0.021, +0.097]. All intervals include zero.
- Weight geometry (post hoc): paired networks end 1.37 (linear) and 1.32 (leaky) times the joint network's weight
  norm apart (medians). Across networks, readout norm, code overlap and readout share explain at most 3% of the
  variance in damage. Across families, the history difference in code overlap tracks the history difference in
  damage in the linear networks (r = 0.26, about 7% of the variance; uncorrected p = 0.037, the largest of six such
  correlations).

## Study 3: a larger confirmation (manuscript Section 6)

Designed from phase 2's saved variance and frozen at 10:07 UTC on 10 October 2026 with 43 numeric predictions; the
same task, histories (still 14 f-only and 8 g-only changes), networks, update conditions and code as phase 2, with 153
new task families (seeds 95001–95153) and initialization seeds 3701 and 3702. The primary endpoint is the damage
difference alone, one per architecture, with two-sided 97.5% intervals (two intervals, 95% simultaneous coverage).
153 families gave an 80% chance of establishing equivalence at ±0.05 if the true difference were zero, at phase 2's
family standard deviation (0.174). The 1,224 networks ran in 16 minutes on six CPU processes.

| | Linear | Leaky ReLU |
|---|---|---|
| Competent pairs | 306 of 306 | 306 of 306 |
| **Damage difference (primary, 97.5%)** | **+0.019 [−0.020, +0.058]** | **+0.018 [−0.017, +0.053]** |
| Decision | inconclusive | inconclusive |
| f-updates (95%) | −0.038 [−0.083, +0.008] | −0.026 [−0.071, +0.018] |
| g-updates (95%) | +0.076 [+0.029, +0.122] | +0.062 [+0.022, +0.103] |
| f minus g (95%) | −0.114 [−0.177, −0.051] | −0.089 [−0.148, −0.029] |
| Pooled with phase 2, 217 families (97.5%, descriptive) | +0.012 [−0.018, +0.043] | +0.015 [−0.013, +0.043] |

- Both intervals contain zero and extend beyond +0.05, by 0.008 and 0.003: equivalence was predicted and narrowly
  missed. The family standard deviations were larger than in phase 2 (0.211 and 0.191 against 0.174 and 0.167).
- A benefit of the independent history larger than 0.020 (linear) or 0.017 (leaky) is excluded.
- The direction split seen post hoc in phase 2 is replicated (prespecified secondary result; 95% intervals without
  multiplicity correction): after the independent history, g-updates do more damage, f-update estimates lie below zero
  with intervals that include zero, and the difference between directions excludes zero. Its source (how often each
  function changed alone, the order of the changes, or the task construction, in which f is drawn first and g
  conditionally on it) cannot be separated with one fixed pair of schedules.
- Everything else replicated phase 2. The changed skill was learned to about 0.0005, the damage was 0.67–0.69, and the
  readout identity held to 2.7 × 10⁻¹⁵ with exact preservation under a frozen readout. With fixed-length steps the
  independent history ended with a lower changed-skill error. 41 of 43 predictions held.
- The verification recomputed every outcome from the raw arrays of all 1,224 networks (136 comparisons, all agree).

## Design history and sample size

- Development (4 families) → amendment (leaky ReLU, chosen after the ReLU networks passed 0 of 8 pairs) →
  confirmation (64 new families, frozen before they were generated). All records were written on 8 October 2026,
  locally, within 30 minutes; manuscript Table 2 marks what was prespecified and what was added later.
- The frozen sample-size rule gave 1,057 families, capped at 64; its multiplier 2.3 is below the t quantile 2.72 that
  the analysis uses. At 64 families, intervals inside ±0.05 had probability 0.3% if the true difference were zero.
- Post hoc planning arithmetic: about 150 families (two intervals) would give an 80% chance of establishing
  equivalence at ±0.05, and about 600 at ±0.025 (manuscript Appendix F). Study 3 used this calculation (153
  families); its larger standard deviations left both intervals just beyond the margin.

## Limits

The networks are small, with one hidden layer and disjoint input weights for the two skills, and linear or nearly so;
the clearly nonlinear (ReLU) networks failed to compose. Only two fixed, unbalanced schedules were compared, with
synthetic functions and one update rule. The phase 2 records were written locally within 30 minutes and study 3
was designed after phase 2's results were known.
