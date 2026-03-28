"""
A2RAG — Query-Adaptive Alpha

Hipotese: Alpha fixo e um compromisso cego. Queries ja proximas do target
sao over-rotated (degradam). Queries distantes sao under-rotated (nao ganham).

Solucao: alpha como funcao da distancia query-target.

Estrategias:
1. Linear: alpha(q) = alpha_max * (1 - cos_sim(q, target))
2. Threshold: alpha(q) = alpha_max if cos_sim(q, target) < threshold else 0
3. Quadratic: alpha(q) = alpha_max * (1 - cos_sim(q, target))^2

Se funcionar, elimina os casos de degradacao enquanto mantem o ganho cross-domain.

Usage: modal run modal_adaptive_alpha.py

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

app = modal.App("a2rag-adaptive-alpha", image=image)
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

ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}

ALPHA_MAXES = [0.1, 0.2, 0.3, 0.5]


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_adaptive_alpha(model_name: str, params_m: int, family: str):
    """Test adaptive alpha strategies for ONE model on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Adaptive Alpha: {model_name} ({params_m}M, {family})")
    print(f"{'='*60}")

    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    model_results = {"model": model_name, "params_m": params_m, "family": family}

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        def eval_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, results, [1, 5, 10])
            return ndcg

        def apply_addition_per_query(q_embs, target, alphas_per_query):
            """Apply addition with per-query alpha."""
            # alphas_per_query: (n_queries,)
            results = q_embs + alphas_per_query.reshape(-1, 1) * target
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            return results / np.maximum(norms, 1e-10)

        # Baseline
        ndcg_base = eval_ndcg(query_embs, corpus_embs)
        base_10 = ndcg_base.get("NDCG@10", 0)
        print(f"    Baseline nDCG@10: {base_10:.4f}")

        # Compute query-target similarities
        q_target_sims = query_embs @ target_emb  # (n_queries,)
        print(f"    Query-target sim: mean={q_target_sims.mean():.3f} std={q_target_sims.std():.3f} min={q_target_sims.min():.3f} max={q_target_sims.max():.3f}")

        ds_results = {
            "baseline": {k: round(v, 4) for k, v in ndcg_base.items()},
            "query_target_sim_stats": {
                "mean": round(float(q_target_sims.mean()), 4),
                "std": round(float(q_target_sims.std()), 4),
                "min": round(float(q_target_sims.min()), 4),
                "max": round(float(q_target_sims.max()), 4),
            },
        }

        for alpha_max in ALPHA_MAXES:
            # === Fixed alpha (reference) ===
            q_fixed = apply_addition_per_query(query_embs, target_emb,
                                               np.full(len(query_ids), alpha_max))
            ndcg_fixed = eval_ndcg(q_fixed, corpus_embs)
            delta_fixed = ndcg_fixed.get("NDCG@10", 0) - base_10

            # === Strategy 1: Linear adaptive ===
            # alpha(q) = alpha_max * (1 - cos_sim(q, target))
            # High sim → low alpha, low sim → high alpha
            alphas_linear = alpha_max * (1 - q_target_sims)
            alphas_linear = np.clip(alphas_linear, 0, alpha_max)
            q_linear = apply_addition_per_query(query_embs, target_emb, alphas_linear)
            ndcg_linear = eval_ndcg(q_linear, corpus_embs)
            delta_linear = ndcg_linear.get("NDCG@10", 0) - base_10

            # === Strategy 2: Threshold ===
            # Only rotate queries below median similarity
            median_sim = np.median(q_target_sims)
            alphas_thresh = np.where(q_target_sims < median_sim, alpha_max, 0.0)
            q_thresh = apply_addition_per_query(query_embs, target_emb, alphas_thresh)
            ndcg_thresh = eval_ndcg(q_thresh, corpus_embs)
            delta_thresh = ndcg_thresh.get("NDCG@10", 0) - base_10

            # === Strategy 3: Quadratic adaptive ===
            # alpha(q) = alpha_max * (1 - cos_sim(q, target))^2
            # Even stronger suppression for close queries
            alphas_quad = alpha_max * (1 - q_target_sims) ** 2
            alphas_quad = np.clip(alphas_quad, 0, alpha_max)
            q_quad = apply_addition_per_query(query_embs, target_emb, alphas_quad)
            ndcg_quad = eval_ndcg(q_quad, corpus_embs)
            delta_quad = ndcg_quad.get("NDCG@10", 0) - base_10

            # === Strategy 4: Inverse (more rotation for close queries) ===
            # Hypothesis: close queries benefit MORE from refinement
            alphas_inv = alpha_max * q_target_sims
            alphas_inv = np.clip(alphas_inv, 0, alpha_max)
            q_inv = apply_addition_per_query(query_embs, target_emb, alphas_inv)
            ndcg_inv = eval_ndcg(q_inv, corpus_embs)
            delta_inv = ndcg_inv.get("NDCG@10", 0) - base_10

            print(f"    α_max={alpha_max}: fixed={delta_fixed:+.4f}  linear={delta_linear:+.4f}"
                  f"  thresh={delta_thresh:+.4f}  quad={delta_quad:+.4f}  inverse={delta_inv:+.4f}")

            # Stats on per-query alphas
            ds_results[f"amax_{alpha_max}"] = {
                "fixed": {"ndcg": {k: round(v, 4) for k, v in ndcg_fixed.items()}, "delta": round(delta_fixed, 4)},
                "linear": {
                    "ndcg": {k: round(v, 4) for k, v in ndcg_linear.items()},
                    "delta": round(delta_linear, 4),
                    "alpha_mean": round(float(alphas_linear.mean()), 4),
                    "alpha_std": round(float(alphas_linear.std()), 4),
                },
                "threshold": {
                    "ndcg": {k: round(v, 4) for k, v in ndcg_thresh.items()},
                    "delta": round(delta_thresh, 4),
                    "pct_rotated": round(float((alphas_thresh > 0).mean()), 4),
                },
                "quadratic": {
                    "ndcg": {k: round(v, 4) for k, v in ndcg_quad.items()},
                    "delta": round(delta_quad, 4),
                    "alpha_mean": round(float(alphas_quad.mean()), 4),
                },
                "inverse": {
                    "ndcg": {k: round(v, 4) for k, v in ndcg_inv.items()},
                    "delta": round(delta_inv, 4),
                    "alpha_mean": round(float(alphas_inv.mean()), 4),
                },
            }

        model_results[ds_name] = ds_results

    safe_name = model_name.replace("/", "_")
    out_path = f"/results/adaptive_alpha/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Query-Adaptive Alpha")
    print("  6 models x 2 datasets x 4 alpha_max x 5 strategies")
    print("=" * 70)

    all_results = list(run_adaptive_alpha.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Best adaptive strategy vs fixed (alpha_max=0.2)")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Fixed':>8} {'Linear':>8} {'Thresh':>8} {'Quad':>8} {'Inv':>8}")
    print("  " + "-" * 84)

    for r in all_results:
        for ds in DATASETS:
            if ds not in r or "amax_0.2" not in r[ds]:
                continue
            a = r[ds]["amax_0.2"]
            print(f"  {r['model']:<30} {ds:<10}"
                  f" {a['fixed']['delta']:>+8.4f}"
                  f" {a['linear']['delta']:>+8.4f}"
                  f" {a['threshold']['delta']:>+8.4f}"
                  f" {a['quadratic']['delta']:>+8.4f}"
                  f" {a['inverse']['delta']:>+8.4f}")

    print("\n  DONE!")
