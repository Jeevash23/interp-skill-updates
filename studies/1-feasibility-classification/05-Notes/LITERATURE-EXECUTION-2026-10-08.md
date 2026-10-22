# AI-07 literature check for experimental execution

The proposed contribution requires a causal account of selective skill updating. Training-order effects, reduced forgetting and functional specialization already have substantial precedent. This focused check supports a small feasibility study, not a novelty claim for a completed paper.

| Primary source | Relevant result and boundary for AI-07 |
|---|---|
| [Kashtan and Alon 2005](https://www.weizmann.ac.il/mcb/alon/sites/mcb.UriAlon/files/spontaneousevolutionofmodularityandnetworkmotifs_0.pdf) | Modularly varying goals produced modular networks under evolutionary search. AI-07 uses gradient training, but changing goals as a route to modularity is established territory. |
| [Yang et al. 2019](https://www.nature.com/articles/s41593-018-0310-2) | Multitask recurrent networks exhibit functional clusters and compositional task representations. Clustering or compositional accuracy alone would not distinguish the proposed learning mechanism. |
| [Bell and Lawrence 2022](https://arxiv.org/abs/2205.13323) | Reordering tasks changes forgetting on synthetic and naturalistic datasets; the work connects ordering to task distance and optimization curvature. AI-07 must address transition difficulty rather than interpreting any order effect as modularity. |
| [Beaulieu et al. 2020](https://arxiv.org/abs/2002.09571) | ANML meta-learns gating that controls activation and plasticity. Selective plasticity itself is not novel; AI-07 asks whether it emerges in ordinary gradient training from an explicitly controlled history. |
| [Li and Hiratani 2025](https://arxiv.org/abs/2502.03350) | A teacher-student analysis and classification experiments connect task similarity and ordering to continual-learning performance. Improved preservation from reordering alone is insufficient evidence of an independently editable computation. |

The searches covered training order, curriculum, modularly varying goals, continual learning and selective plasticity. They do not establish an exhaustive absence of the exact proposed experiment. A mechanism claim would need fixed causal interventions, matched comparison conditions and fresh task families. A feasibility failure should instead be reported within the small MLP and finite-task setting.
