"""
A²RAG — Phase 3: Conceptor and Generative Operations on BEIR.

Operations requiring specialized frameworks:
  1. Boolean Conceptors — matrix-based concept representation + AND/OR/NOT
  2. Hippocampal Vector Addition — generative memory-inspired pattern completion
  3. Conceptor Negation — NOT(concept) as retrieval filter
  4. Conceptor Blending — soft AND of multiple concepts

These are the most experimental operations from the original 17.
6 models × 2 datasets, each model in its own container.

Usage: modal run --detach modal_17ops_phase3.py

Author: Renato Aparecido Gomes
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0",
        "beir",
        "torch",
        "numpy",
        "scipy",
        "pytrec_eval",
        "datasets",
        "faiss-cpu",
    )
)

app = modal.App("a2rag-17ops-phase3", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", 22, "distilled"),
    ("BAAI/bge-small-en-v1.5", 33, "contrastive"),
    ("all-mpnet-base-v2", 109, "trained-1B-pairs"),
    ("BAAI/bge-base-en-v1.5", 109, "contrastive"),
    ("intfloat/e5-small-v2", 33, "instruction-tuned"),
    ("thenlper/gte-small", 33, "general-text"),
]

DATASETS = ["scifact", "arguana"]


# ═══════════════════════════════════════════════════════════════════
# BOOLEAN CONCEPTORS
# ═══════════════════════════════════════════════════════════════════

class Conceptor:
    """Boolean Conceptor: matrix-based concept representation.

    A conceptor C is a d×d positive semi-definite matrix that "filters"
    embeddings to retain only the concept-aligned components.

    Based on Jaeger (2014) "Controlling Recurrent Neural Networks by
    Conceptors" — adapted for embedding spaces.
    """

    def __init__(self, embeddings, aperture=10.0):
        """Create conceptor from example embeddings.

        C = R(R + α⁻²I)⁻¹  where R = (1/n) Σ xᵢxᵢᵀ (correlation matrix)
        """
        import numpy as np
        n, d = embeddings.shape
        # Correlation matrix (use a subset for speed)
        if n > 500:
            idx = np.random.choice(n, 500, replace=False)
            R = embeddings[idx].T @ embeddings[idx] / 500
        else:
            R = embeddings.T @ embeddings / n

        # Conceptor: C = R(R + α⁻²I)⁻¹
        alpha_sq_inv = 1.0 / (aperture ** 2)
        self.C = R @ np.linalg.inv(R + alpha_sq_inv * np.eye(d))
        self.d = d

    def apply(self, embeddings):
        """Filter embeddings through this conceptor."""
        import numpy as np
        result = embeddings @ self.C.T
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-10)

    def NOT(self):
        """Boolean NOT: ¬C = I - C"""
        import numpy as np
        neg = Conceptor.__new__(Conceptor)
        neg.C = np.eye(self.d) - self.C
        neg.d = self.d
        return neg

    @staticmethod
    def AND(c1, c2):
        """Boolean AND: C₁ ∧ C₂ (soft intersection)"""
        import numpy as np
        d = c1.d
        # C₁ ∧ C₂ = (C₁⁻¹ + C₂⁻¹ - I)⁻¹ (with regularization)
        eps = 1e-4 * np.eye(d)
        try:
            inv1 = np.linalg.inv(c1.C + eps)
            inv2 = np.linalg.inv(c2.C + eps)
            result_inv = inv1 + inv2 - np.eye(d)
            C_and = np.linalg.inv(result_inv + eps)
        except np.linalg.LinAlgError:
            C_and = 0.5 * (c1.C + c2.C)

        out = Conceptor.__new__(Conceptor)
        out.C = C_and
        out.d = d
        return out

    @staticmethod
    def OR(c1, c2):
        """Boolean OR: C₁ ∨ C₂ = ¬(¬C₁ ∧ ¬C₂)"""
        not_c1 = c1.NOT()
        not_c2 = c2.NOT()
        not_and = Conceptor.AND(not_c1, not_c2)
        return not_and.NOT()


def op_conceptor_filter(query_embs, concept_corpus_embs, aperture=10.0):
    """Apply conceptor built from concept-related docs as a filter."""
    C = Conceptor(concept_corpus_embs, aperture)
    return C.apply(query_embs)


def op_conceptor_not(query_embs, concept_corpus_embs, aperture=10.0):
    """Apply NOT(conceptor) — filter OUT the concept."""
    C = Conceptor(concept_corpus_embs, aperture)
    return C.NOT().apply(query_embs)


def op_conceptor_and(query_embs, corpus_embs_1, corpus_embs_2, aperture=10.0):
    """Apply AND of two conceptors — intersection of concepts."""
    C1 = Conceptor(corpus_embs_1, aperture)
    C2 = Conceptor(corpus_embs_2, aperture)
    C_and = Conceptor.AND(C1, C2)
    return C_and.apply(query_embs)


def op_conceptor_or(query_embs, corpus_embs_1, corpus_embs_2, aperture=10.0):
    """Apply OR of two conceptors — union of concepts."""
    C1 = Conceptor(corpus_embs_1, aperture)
    C2 = Conceptor(corpus_embs_2, aperture)
    C_or = Conceptor.OR(C1, C2)
    return C_or.apply(query_embs)


# ═══════════════════════════════════════════════════════════════════
# HIPPOCAMPAL VECTOR ADDITION
# ═══════════════════════════════════════════════════════════════════

def op_hippocampal_addition(query_embs, corpus_embs, target_emb, k_neighbors=5, alpha=0.1):
    """Hippocampal pattern completion: use nearest neighbors as memory.

    Inspired by hippocampal indexing theory:
    1. For each query, find k nearest corpus docs (memory retrieval)
    2. Compute the centroid of these neighbors (pattern completion)
    3. Blend query with completed pattern, biased toward target

    This simulates how the hippocampus completes partial patterns
    using stored memories.
    """
    import numpy as np

    sims = query_embs @ corpus_embs.T  # (n_q, n_c)
    results = np.zeros_like(query_embs)

    for i in range(len(query_embs)):
        # Retrieve k nearest memories
        top_k = np.argsort(sims[i])[::-1][:k_neighbors]
        memories = corpus_embs[top_k]

        # Pattern completion: centroid of memories
        pattern = memories.mean(axis=0)

        # Bias completion toward target concept
        biased_pattern = (1 - alpha) * pattern + alpha * target_emb
        biased_pattern = biased_pattern / (np.linalg.norm(biased_pattern) + 1e-10)

        # Blend original query with completed pattern
        result = 0.7 * query_embs[i] + 0.3 * biased_pattern
        results[i] = result

    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_hippocampal_reconstruction(query_embs, corpus_embs, k_neighbors=10):
    """Hippocampal reconstruction: reconstruct query from memory traces.

    Uses a linear combination of nearest neighbors weighted by similarity.
    This is essentially kernel regression in embedding space.
    """
    import numpy as np

    sims = query_embs @ corpus_embs.T
    results = np.zeros_like(query_embs)

    for i in range(len(query_embs)):
        top_k = np.argsort(sims[i])[::-1][:k_neighbors]
        weights = sims[i, top_k]
        # Softmax weights
        weights = np.exp(weights - weights.max())
        weights = weights / (weights.sum() + 1e-10)
        results[i] = weights @ corpus_embs[top_k]

    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def benchmark_model(model_name: str, params_m: int, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*70}")
    print(f"  Phase 3: {model_name} ({params_m}M, {family})")
    print(f"{'='*70}")

    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    all_results = {"model": model_name, "params_M": params_m, "family": family}

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-data-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                     for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        print(f"    Encoding...")
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                             normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                            show_progress_bar=False))

        # Concepts
        if ds_name == "scifact":
            target_text = "clinical medicine and patient outcomes"
            exclude_text = "methodology and statistical analysis"
            concept_queries_1 = ["clinical trials", "patient treatment", "drug efficacy",
                                 "disease outcomes", "medical intervention"]
            concept_queries_2 = ["gene expression", "protein structure", "DNA sequencing",
                                 "molecular biology", "cellular mechanism"]
        else:
            target_text = "economic policy and market regulation"
            exclude_text = "moral and ethical reasoning"
            concept_queries_1 = ["tax policy", "market regulation", "fiscal stimulus",
                                 "trade agreement", "economic growth"]
            concept_queries_2 = ["human rights", "ethical framework", "moral philosophy",
                                 "social justice", "cultural values"]

        target_emb = model.encode(target_text, normalize_embeddings=True)

        # Build concept corpora: find docs most similar to concept queries
        cq1_embs = np.array(model.encode(concept_queries_1, normalize_embeddings=True))
        cq2_embs = np.array(model.encode(concept_queries_2, normalize_embeddings=True))

        # Top-50 docs for each concept cluster
        sims1 = cq1_embs.mean(axis=0) @ corpus_embs.T
        sims2 = cq2_embs.mean(axis=0) @ corpus_embs.T
        concept_idx_1 = np.argsort(sims1)[::-1][:50]
        concept_idx_2 = np.argsort(sims2)[::-1][:50]
        concept_embs_1 = corpus_embs[concept_idx_1]
        concept_embs_2 = corpus_embs[concept_idx_2]

        def eval_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        def jaccard_shift(q_orig, q_mod, c_embs, k=10):
            s1 = q_orig @ c_embs.T
            s2 = q_mod @ c_embs.T
            jaccards = []
            for i in range(len(q_orig)):
                t1 = set(np.argsort(s1[i])[::-1][:k])
                t2 = set(np.argsort(s2[i])[::-1][:k])
                jaccards.append(len(t1 & t2) / len(t1 | t2) if t1 | t2 else 1.0)
            return float(np.mean(jaccards))

        baseline = eval_ndcg(query_embs, corpus_embs)
        print(f"    Baseline: {baseline:.4f}")
        ds_results = {"baseline_ndcg10": baseline}

        # ── Conceptor Filter ──
        for aperture in [1.0, 10.0, 100.0]:
            modified = op_conceptor_filter(query_embs, concept_embs_1, aperture)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"conceptor_filter_a{aperture}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
            }
            print(f"    Conceptor Filter a={aperture}: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        # ── Conceptor NOT ──
        for aperture in [1.0, 10.0, 100.0]:
            modified = op_conceptor_not(query_embs, concept_embs_2, aperture)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"conceptor_not_a{aperture}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
            }
            print(f"    Conceptor NOT a={aperture}: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        # ── Conceptor AND ──
        modified = op_conceptor_and(query_embs, concept_embs_1, concept_embs_2, 10.0)
        ndcg = eval_ndcg(modified, corpus_embs)
        jacc = jaccard_shift(query_embs, modified, corpus_embs)
        ds_results["conceptor_and"] = {
            "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
        }
        print(f"    Conceptor AND: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        # ── Conceptor OR ──
        modified = op_conceptor_or(query_embs, concept_embs_1, concept_embs_2, 10.0)
        ndcg = eval_ndcg(modified, corpus_embs)
        jacc = jaccard_shift(query_embs, modified, corpus_embs)
        ds_results["conceptor_or"] = {
            "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
        }
        print(f"    Conceptor OR: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        # ── Hippocampal Addition ──
        for alpha in [0.1, 0.2, 0.3]:
            modified = op_hippocampal_addition(query_embs, corpus_embs, target_emb,
                                               k_neighbors=5, alpha=alpha)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"hippo_add_a{alpha}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
            }
            print(f"    Hippo Add α={alpha}: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        # ── Hippocampal Reconstruction ──
        for k in [5, 10, 20]:
            modified = op_hippocampal_reconstruction(query_embs, corpus_embs, k_neighbors=k)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"hippo_recon_k{k}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline, 4), "jaccard": jacc
            }
            print(f"    Hippo Recon k={k}: {ndcg:.4f} Δ={ndcg-baseline:+.4f} J={jacc:.3f}")

        all_results[ds_name] = ds_results

    # Save
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/17ops_phase3/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved to {out_path}")
    return all_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A²RAG Phase 3: Conceptors + Hippocampal Operations")
    print("  6 models × 2 datasets × 6 operation types")
    print("=" * 70)

    results = list(benchmark_model.map(
        [m[0] for m in MODELS], [m[1] for m in MODELS], [m[2] for m in MODELS],
    ))

    print(f"\n{'='*120}")
    print(f"  SUMMARY — Phase 3")
    print(f"{'='*120}")
    print(f"{'Model':<25} {'DS':<8} {'Base':>7} {'CFilt':>7} {'CNOT':>7} "
          f"{'CAND':>7} {'COR':>7} {'HAdd':>7} {'HRec':>7}")
    print("-" * 100)
    for r in results:
        for ds in DATASETS:
            d = r[ds]
            short = r["model"].split("/")[-1][:24]
            print(f"{short:<25} {ds:<8} "
                  f"{d['baseline_ndcg10']:>7.4f} "
                  f"{d.get('conceptor_filter_a10.0',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('conceptor_not_a10.0',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('conceptor_and',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('conceptor_or',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('hippo_add_a0.1',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('hippo_recon_k10',{}).get('ndcg10',0):>7.4f}")

    print(f"\n  Download: modal volume get a2rag-results 17ops_phase3/ ./results_modal/17ops_phase3/")
