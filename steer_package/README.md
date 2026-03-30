# STEER

**Semantic Transformation for Embedding-space Exploration in Retrieval**

A retrieval primitive for controllable semantic navigation in embedding spaces. No retraining, no fine-tuning, works with any embedding model.

> Paper: [STEER: An Honest Evaluation of Sixteen Algebraic Operations](https://arxiv.org/abs/XXXX.XXXXX)

## Install

```bash
pip install steer-retrieval
```

## Quick Start

```python
from steer import Steerer

# Initialize with any sentence-transformer model
steerer = Steerer(model_name="all-MiniLM-L6-v2")

# Index your corpus
steerer.index(["doc1 text...", "doc2 text...", ...])

# Baseline search
results = steerer.search("VEGF inhibition in retinal disease")

# Steered search: explore adjacent domain
results = steerer.search(
    "VEGF inhibition in retinal disease",
    target="oncology drug mechanisms",
    alpha=0.1,
)

# Contrastive: push toward positive, away from negative
results = steerer.contrastive_search(
    "VEGF inhibition in retinal disease",
    positive_target="oncology drug mechanisms",
    negative_target="basic laboratory research",
    alpha=0.1,
    beta=0.2,
)

# Gradient walk: see how results change across alpha values
walk = steerer.gradient_walk(
    "VEGF inhibition in retinal disease",
    target="oncology drug mechanisms",
)
for step in walk:
    print(f"alpha={step['alpha']}: {step['results'][0]['id']}")
```

## Operations

STEER provides 16 operations built from a single primitive: `normalize(q + alpha * t)`.

| Operation | Function | Description |
|-----------|----------|-------------|
| Rotate Toward | `rotate_toward(q, t, alpha)` | Move query toward target domain |
| Rotate Away | `rotate_away(q, t, alpha)` | Move query away (bias correction) |
| Contrastive | `contrastive(q, t_pos, t_neg, alpha, beta)` | Push/pull simultaneously |
| Amplify | `amplify(q, centroid, alpha)` | Intensify specificity |
| Diffuse | `diffuse(q, centroid, alpha)` | Generalize, increase diversity |
| Multi-View | `multi_view(q, targets, corpus, alpha)` | Parallel search + RRF fusion |
| Gradient Walk | `gradient_walk(q, t, corpus)` | Degradation curve for calibration |
| Consensus | `consensus(q, targets, alpha)` | Mean of multiple targets |
| Orbit | `orbit(q, targets, corpus, alpha)` | N separate result sets |
| Bridge | `bridge(q, source, dest, alpha)` | Domain transfer |
| Triangulate | `triangulate(q, t1, t2, a1, a2)` | Sequential 2-target steering |
| Isotropy Correct | `isotropy_correct(embeddings, k)` | Top-k PCA removal |
| Adaptive Alpha | `adaptive_alpha(q, t, alpha_max)` | Auto-calibrate per query |
| RRF Fusion | `rrf_fusion(scores_list, k)` | Reciprocal rank fusion |

## Key Findings

- **Contrastive asymmetric** (alpha=0.1, beta=0.2) produces the strongest results: +3.5% nDCG
- **Rotate Away** unexpectedly improves retrieval: +3.6% nDCG (bias correction)
- **Adaptive alpha** reduces degradation by 88-93%
- **Multi-vector RRF** eliminates 100% of degradation
- **Classifier** with per-query LLM targets achieves F1=0.91 as automatic router

## Citation

```bibtex
@article{gomes2026steer,
  title={STEER: Semantic Transformation for Embedding-space Exploration in Retrieval --- An Honest Evaluation of Sixteen Algebraic Operations},
  author={Gomes, Renato Aparecido},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
