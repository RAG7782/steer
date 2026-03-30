# Email para Nils Reimers (Cohere)

**Para:** nils@cohere.com (ou via LinkedIn/Twitter DM)
**Assunto:** Controllable semantic steering for dense retrieval — built on Sentence-Transformers

---

Hi Nils,

I built a controllable steering primitive for dense retrieval on top of the Sentence-Transformers models you created, and I thought you'd be interested in the results.

**The idea:** `normalize(q + alpha * target)` — one-line operation that steers a query toward (or away from) a semantic direction, with no retraining.

**What I found (across 6 ST models, 5 BEIR datasets, bootstrap significance):**

- Contrastive steer (push toward + pull away) achieves +3.5% nDCG on TREC-COVID
- Rotating *away* from generic targets improves retrieval by +3.6% (bias correction — the target is a noise attractor)
- An adaptive stack (isotropy correction + adaptive alpha + multi-vector RRF) eliminates all degradation
- A classifier with per-query LLM targets achieves F1=0.91 as automatic operation router

The paper is at [arXiv link]. The code is at [GitHub link].

I'm particularly interested in how this could fit into Cohere's Embed + Rerank pipeline as a controllable "exploration mode." Happy to discuss if useful.

Best,
Renato Aparecido Gomes
São Paulo, Brazil
