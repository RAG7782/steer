"""
A²RAG — Phase 1: Empirical Benchmark of 7 Algebraic Operations on BEIR.

Tests operations beyond rotation and subtraction, with the same rigor:
- 6 embedding models × 2 datasets (SciFact, ArguAna) × 7 operations
- Each (model, dataset) pair runs in its own container
- Metrics: nDCG@10, Jaccard shift, isotropy impact

Operations tested:
  1. SLERP (spherical linear interpolation) — the "correct" rotation
  2. Scaled Subtraction — parameterized projection removal
  3. Directional Scaling — amplify/attenuate along a direction
  4. Weighted Composition — multi-concept blending
  5. Sequential Composition — chained operations (rot→sub, sub→rot)
  6. Vector Addition — simple concept injection
  7. Gumbel-Softmax k-NN — probabilistic reranking

Usage:
    modal run --detach modal_17ops_phase1.py        # all operations
    modal run modal_17ops_phase1.py::run_single_op  # test one operation

Author: Renato Aparecido Gomes
"""

import modal
import json
import os

# ─── Modal Infrastructure ───────────────────────────────────────────

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

app = modal.App("a2rag-17ops-phase1", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)


# ─── Models to test ─────────────────────────────────────────────────

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
# ALGEBRAIC OPERATIONS — Each returns modified query embeddings
# ═══════════════════════════════════════════════════════════════════

def op_nlerp(query_embs, target_emb, alpha=0.1):
    """NLERP: Normalized linear interpolation (our baseline rotation)."""
    import numpy as np
    results = (1 - alpha) * query_embs + alpha * target_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return results / norms


def op_slerp(query_embs, target_emb, alpha=0.1):
    """SLERP: Spherical linear interpolation (geometrically correct)."""
    import numpy as np
    # Normalize inputs
    q_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    t_norm = target_emb / np.linalg.norm(target_emb)

    # Compute angle between each query and target
    dots = np.clip(q_norm @ t_norm, -1.0, 1.0)
    omegas = np.arccos(dots)  # shape: (n_queries,)

    results = np.zeros_like(query_embs)
    for i in range(len(query_embs)):
        omega = omegas[i]
        if omega < 1e-6:  # nearly parallel
            results[i] = q_norm[i]
        else:
            sin_omega = np.sin(omega)
            results[i] = (np.sin((1 - alpha) * omega) / sin_omega) * q_norm[i] + \
                          (np.sin(alpha * omega) / sin_omega) * t_norm
    # Normalize output
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_subtraction(query_embs, exclude_emb):
    """Orthogonal subtraction: remove concept projection."""
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_scaled_subtraction(query_embs, exclude_emb, beta=0.5):
    """Scaled subtraction: parameterized projection removal (partial)."""
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - beta * (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_directional_scaling(query_embs, direction_emb, gamma=1.5):
    """Directional scaling: amplify component along a direction."""
    import numpy as np
    d_norm = direction_emb / (np.linalg.norm(direction_emb) + 1e-10)
    proj = (query_embs @ d_norm).reshape(-1, 1)
    # Decompose: parallel + orthogonal
    parallel = proj * d_norm
    orthogonal = query_embs - parallel
    # Scale the parallel component
    results = orthogonal + gamma * parallel
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_addition(query_embs, concept_emb, alpha=0.1):
    """Vector addition: inject concept direction."""
    import numpy as np
    results = query_embs + alpha * concept_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_weighted_composition(query_embs, target_emb, exclude_emb,
                            w_rot=0.5, w_sub=0.5, alpha=0.1):
    """Weighted composition: blend rotation and subtraction."""
    import numpy as np
    rotated = op_nlerp(query_embs, target_emb, alpha)
    subtracted = op_subtraction(query_embs, exclude_emb)
    results = w_rot * rotated + w_sub * subtracted
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_sequential_rot_sub(query_embs, target_emb, exclude_emb, alpha=0.1):
    """Sequential: rotate first, then subtract."""
    rotated = op_nlerp(query_embs, target_emb, alpha)
    return op_subtraction(rotated, exclude_emb)


def op_sequential_sub_rot(query_embs, target_emb, exclude_emb, alpha=0.1):
    """Sequential: subtract first, then rotate."""
    subtracted = op_subtraction(query_embs, exclude_emb)
    return op_nlerp(subtracted, target_emb, alpha)


def op_gumbel_softmax_rerank(query_embs, corpus_embs, doc_ids, tau=0.5, top_k=100):
    """Gumbel-Softmax k-NN: probabilistic reranking with temperature.

    Instead of hard top-k, applies Gumbel noise + softmax to similarity scores.
    Returns modified retrieval results (not modified embeddings).
    """
    import numpy as np
    sims = query_embs @ corpus_embs.T
    # Add Gumbel noise
    np.random.seed(42)
    gumbel_noise = -np.log(-np.log(np.random.uniform(size=sims.shape) + 1e-20) + 1e-20)
    noisy_sims = (sims + gumbel_noise * 0.1) / tau
    # Softmax over top candidates
    results = {}
    for i in range(len(query_embs)):
        top_idx = np.argsort(noisy_sims[i])[::-1][:top_k]
        # Use softmax scores as retrieval scores
        exp_scores = np.exp(noisy_sims[i, top_idx] - noisy_sims[i, top_idx].max())
        softmax_scores = exp_scores / exp_scores.sum()
        results[i] = {doc_ids[idx]: float(softmax_scores[j]) for j, idx in enumerate(top_idx)}
    return results


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTION — runs on Modal
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def benchmark_model(model_name: str, params_m: int, family: str):
    """Run all 7 operations for one model across both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*70}")
    print(f"  Model: {model_name} ({params_m}M, {family})")
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

        # Encode
        print(f"    Encoding {len(doc_texts)} docs + {len(query_texts)} queries...")
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                             normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                            show_progress_bar=False))

        # Concept embeddings for operations
        if ds_name == "scifact":
            target_text = "clinical medicine and patient outcomes"
            exclude_text = "methodology and statistical analysis"
        else:  # arguana
            target_text = "economic policy and market regulation"
            exclude_text = "moral and ethical reasoning"

        target_emb = model.encode(target_text, normalize_embeddings=True)
        exclude_emb = model.encode(exclude_text, normalize_embeddings=True)

        # Helper: evaluate nDCG from embeddings
        def eval_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        # Helper: Jaccard shift (how much results change)
        def jaccard_shift(q_embs_orig, q_embs_modified, c_embs, k=10):
            sims_orig = q_embs_orig @ c_embs.T
            sims_mod = q_embs_modified @ c_embs.T
            jaccards = []
            for i in range(len(q_embs_orig)):
                orig_top = set(np.argsort(sims_orig[i])[::-1][:k])
                mod_top = set(np.argsort(sims_mod[i])[::-1][:k])
                inter = len(orig_top & mod_top)
                union = len(orig_top | mod_top)
                jaccards.append(inter / union if union > 0 else 1.0)
            return float(np.mean(jaccards))

        # ── Baseline ──
        baseline_ndcg = eval_ndcg(query_embs, corpus_embs)
        print(f"    Baseline nDCG@10: {baseline_ndcg:.4f}")

        ds_results = {"baseline_ndcg10": baseline_ndcg}

        # ── 1. NLERP (reference) ──
        for alpha in [0.1, 0.2]:
            modified = op_nlerp(query_embs, target_emb, alpha)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"nlerp_a{alpha}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    NLERP α={alpha}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 2. SLERP ──
        for alpha in [0.1, 0.2]:
            modified = op_slerp(query_embs, target_emb, alpha)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"slerp_a{alpha}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    SLERP α={alpha}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 3. Subtraction (reference) ──
        modified = op_subtraction(query_embs, exclude_emb)
        ndcg = eval_ndcg(modified, corpus_embs)
        jacc = jaccard_shift(query_embs, modified, corpus_embs)
        ds_results["subtraction"] = {
            "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
            "jaccard": jacc
        }
        print(f"    Subtraction: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 4. Scaled Subtraction ──
        for beta in [0.25, 0.5, 0.75]:
            modified = op_scaled_subtraction(query_embs, exclude_emb, beta)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"scaled_sub_b{beta}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    Scaled Sub β={beta}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 5. Directional Scaling ──
        for gamma in [0.5, 1.5, 2.0]:
            modified = op_directional_scaling(query_embs, target_emb, gamma)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"dir_scale_g{gamma}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    Dir Scale γ={gamma}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 6. Vector Addition ──
        for alpha in [0.1, 0.2]:
            modified = op_addition(query_embs, target_emb, alpha)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"addition_a{alpha}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    Addition α={alpha}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 7. Weighted Composition ──
        for w_rot, w_sub in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
            modified = op_weighted_composition(query_embs, target_emb, exclude_emb,
                                               w_rot, w_sub, alpha=0.1)
            ndcg = eval_ndcg(modified, corpus_embs)
            jacc = jaccard_shift(query_embs, modified, corpus_embs)
            ds_results[f"composition_r{w_rot}_s{w_sub}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
                "jaccard": jacc
            }
            print(f"    Composition r={w_rot}/s={w_sub}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 8. Sequential: Rot→Sub ──
        modified = op_sequential_rot_sub(query_embs, target_emb, exclude_emb, alpha=0.1)
        ndcg = eval_ndcg(modified, corpus_embs)
        jacc = jaccard_shift(query_embs, modified, corpus_embs)
        ds_results["seq_rot_sub"] = {
            "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
            "jaccard": jacc
        }
        print(f"    Seq Rot→Sub: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 9. Sequential: Sub→Rot ──
        modified = op_sequential_sub_rot(query_embs, target_emb, exclude_emb, alpha=0.1)
        ndcg = eval_ndcg(modified, corpus_embs)
        jacc = jaccard_shift(query_embs, modified, corpus_embs)
        ds_results["seq_sub_rot"] = {
            "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
            "jaccard": jacc
        }
        print(f"    Seq Sub→Rot: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f} J={jacc:.3f}")

        # ── 10. Gumbel-Softmax k-NN ──
        for tau in [0.3, 0.5, 1.0]:
            gumbel_results = op_gumbel_softmax_rerank(query_embs, corpus_embs, doc_ids, tau)
            # Convert to format for evaluation
            gumbel_dict = {}
            for i, qid in enumerate(query_ids):
                if i in gumbel_results:
                    gumbel_dict[qid] = gumbel_results[i]
            ndcg_dict, _, _, _ = evaluator.evaluate(qrels, gumbel_dict, [10])
            ndcg = ndcg_dict.get("NDCG@10", 0)
            ds_results[f"gumbel_tau{tau}"] = {
                "ndcg10": ndcg, "delta": round(ndcg - baseline_ndcg, 4),
            }
            print(f"    Gumbel τ={tau}: nDCG={ndcg:.4f} Δ={ndcg-baseline_ndcg:+.4f}")

        all_results[ds_name] = ds_results

    # Save per-model results
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/17ops_phase1/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved to {out_path}")

    return all_results


@app.local_entrypoint()
def main():
    """Run all 6 models in parallel — each in its own container."""
    print("=" * 70)
    print("  A²RAG Phase 1: 7 Algebraic Operations × 6 Models × 2 Datasets")
    print("  Total: ~168 operation-dataset-model combinations")
    print("=" * 70)

    results = list(benchmark_model.map(
        [m[0] for m in MODELS],
        [m[1] for m in MODELS],
        [m[2] for m in MODELS],
    ))

    # Summary table
    print(f"\n{'='*130}")
    print(f"  SUMMARY — Phase 1: Key Operations")
    print(f"{'='*130}")
    print(f"{'Model':<25} {'DS':<8} {'Base':>7} {'NLERP':>7} {'SLERP':>7} "
          f"{'Sub':>7} {'ScSub':>7} {'DirSc':>7} {'Add':>7} {'Comp':>7} "
          f"{'R→S':>7} {'S→R':>7} {'Gumb':>7}")
    print("-" * 130)

    for r in results:
        for ds in DATASETS:
            d = r[ds]
            short = r["model"].split("/")[-1][:24]
            print(f"{short:<25} {ds:<8} "
                  f"{d['baseline_ndcg10']:>7.4f} "
                  f"{d.get('nlerp_a0.1',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('slerp_a0.1',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('subtraction',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('scaled_sub_b0.5',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('dir_scale_g1.5',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('addition_a0.1',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('composition_r0.5_s0.5',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('seq_rot_sub',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('seq_sub_rot',{}).get('ndcg10',0):>7.4f} "
                  f"{d.get('gumbel_tau0.5',{}).get('ndcg10',0):>7.4f}")

    print(f"\n  Download: modal volume get a2rag-results 17ops_phase1/ ./results_modal/17ops_phase1/")
