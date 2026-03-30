# Email para Han Xiao (Jina AI)

**Para:** han@jina.ai (ou via Twitter @haboroshan)
**Assunto:** Post-hoc semantic steering complements instruction-tuned embeddings

---

Hi Han,

Your work on instruction-tuned embeddings (jina-embeddings-v2/v3) inspired me to investigate the complementary approach: post-hoc steering of query embeddings *after* encoding.

**Key finding:** `normalize(q + alpha * target)` provides controllable cross-domain exploration that works with *any* embedding model, including instruction-tuned ones. But the interesting part is how different model architectures respond:

- Instruction-tuned models (like E5): benefit from contrastive steer (+3.5% nDCG on biomedical benchmarks)
- Distilled models (MiniLM): tolerate extreme alpha values without degradation (never <95% baseline even at alpha=0.5)
- Contrastive models (BGE): benefit most from rotating *away* from targets (+3.6% — targets act as noise attractors)

The adaptive stack (isotropy correction + adaptive alpha + multi-vector RRF) makes this safe for production. And a classifier with per-query LLM targets achieves F1=0.91 as automatic router.

Paper: [arXiv link]. Code: [GitHub link].

This is complementary to instruction tuning — steering post-hoc does things that instruction prefixes can't (e.g., contrastive push/pull, gradient walk, orbit). Would be interesting to discuss whether this could integrate with Jina's embedding pipeline.

Best,
Renato Aparecido Gomes
