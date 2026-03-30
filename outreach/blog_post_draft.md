# Blog Post Draft — Medium/Substack

**Title:** I Tested 16 Ways to Steer Embedding Search. Here's What Actually Works.

**Subtitle:** STEER: A retrieval primitive for controllable semantic navigation — no retraining needed.

---

What if you could point your search in a direction — without retraining your model?

I spent two weeks testing this idea systematically. 16 algebraic operations. 6 embedding models. 5 BEIR datasets. 146 result files. Bootstrap significance testing. Here's what I found.

## The Primitive

One line of code:

```python
q_steered = normalize(q + alpha * target)
```

That's it. Take a query embedding, add a scaled target direction, normalize. The query now "points" slightly toward the target domain.

## What Works (and what surprised me)

### 1. Moving AWAY from the target improves search

This was the biggest surprise. I expected "rotate toward target = better results." Instead, rotating *away* from generic targets like "clinical medicine" produced the largest improvement: +3.6% nDCG on SciFact with BGE-base.

Why? Generic targets are **noise attractors**. They point toward a dense, generic region in embedding space. Moving away from that noise sharpens the query's discriminative signal.

### 2. Contrastive steer: push + pull

`normalize(q + 0.1*positive - 0.2*negative)` — push toward what you want, pull away from what you don't. The asymmetric config (weak positive, strong negative) produces the best results overall: +3.5% on TREC-COVID.

The negative component is more powerful than the positive. Removing noise beats adding signal.

### 3. An automatic router decides when to steer

With per-query LLM-generated targets (Qwen2.5-3B, ~$0.001/query), a random forest classifier achieves F1=0.91 at predicting which queries benefit from steering. 91% accuracy — near perfect.

### 4. Some operations are dangerous

Chaining N steerings sequentially destroys results. N=10 steps of alpha=0.1 produces -12% to -57% nDCG. The intermediate normalization accumulates geometric distortion. Don't chain.

## The Adaptive Stack (Production-Ready)

Three components compose into a safe, effective system:

1. **Adaptive alpha:** `alpha(q) = alpha_max * (1 - cos_sim(q, target))^2` — queries close to target get minimal rotation
2. **Multi-vector RRF:** search both original q and steered T(q), fuse results — preserves baseline while surfacing new docs
3. **Per-query targets:** LLM generates specific targets per query instead of generic ones

This stack achieves +2% to +5% nDCG on biomedical datasets with zero degradation on others.

## What I Learned

STEER is not a replacement for better embeddings. It's a **navigation instrument**. Like a compass doesn't change the terrain, STEER doesn't change the embedding space — it lets you choose where to look within it.

The practical value is for domain experts who need cross-domain exploration: drug repurposing researchers, patent analysts, legal professionals. The person who searches "VEGF inhibition in retinal disease" and needs to find oncology papers about the same pathway — that's where STEER shines.

**Paper:** [arXiv link]
**Code:** `pip install steer-retrieval` | [GitHub link]

---

*Renato Aparecido Gomes is an independent researcher in São Paulo, Brazil, working on semantic search and retrieval-augmented generation.*
