"""
STEER — Full Significance Testing (5 datasets)

Extends bootstrap significance to all 5 BEIR datasets (not just scifact+arguana).
Tests 5 strategies: addition_uniform, adaptive_alpha, multivector_generic,
full_stack, and adaptive_stack (no top-k).

6 models x 5 datasets x 5 strategies x 1000 bootstrap = comprehensive.

Usage: modal run modal_significance_full.py
Author: Renato Aparecido Gomes
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0", "beir", "torch", "numpy",
        "scipy", "pytrec_eval", "datasets", "faiss-cpu", "scikit-learn",
    )
)

app = modal.App("steer-significance-full", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("all-mpnet-base-v2", "trained"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("intfloat/e5-small-v2", "instruction"),
    ("thenlper/gte-small", "general"),
]

DATASETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
    "nfcorpus": "clinical nutrition interventions",
    "fiqa": "macroeconomic policy impacts",
    "trec-covid": "COVID-19 clinical treatment protocols",
}

ALPHA = 0.2
RRF_K = 60
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_significance(model_name: str, family: str):
    import numpy as np
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Significance Full: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def remove_top_k(embs, k=1):
        pca = PCA()
        pca.fit(embs)
        result = embs.copy()
        comps = pca.components_[:k]
        for c in comps:
            result -= (result @ c).reshape(-1, 1) * c
        return normalize_rows(result), comps

    def rrf_fusion(scores_list, k=60):
        n = scores_list[0].shape[0]
        rrf = np.zeros(n)
        for s in scores_list:
            ranks = np.argsort(np.argsort(-s)) + 1
            rrf += 1.0 / (k + ranks)
        return rrf

    def per_query_ndcg(query_ids, doc_ids, scores_matrix, qrels_dict, k=10):
        per_q = []
        for i, qid in enumerate(query_ids):
            if qid not in qrels_dict: continue
            top_idx = np.argsort(scores_matrix[i])[::-1][:k]
            dcg = sum(qrels_dict[qid].get(doc_ids[idx], 0) / np.log2(r + 2) for r, idx in enumerate(top_idx))
            rels = sorted(qrels_dict[qid].values(), reverse=True)[:k]
            idcg = sum(rel / np.log2(j + 2) for j, rel in enumerate(rels))
            per_q.append(dcg / max(idcg, 1e-10))
        return np.array(per_q)

    def bootstrap_ci(base_pq, method_pq, n=N_BOOTSTRAP):
        rng = np.random.RandomState(42)
        deltas = method_pq - base_pq
        n_q = len(deltas)
        boot = sorted([np.mean(deltas[rng.choice(n_q, n_q, replace=True)]) for _ in range(n)])
        ci_lo, ci_hi = boot[int(0.025 * n)], boot[int(0.975 * n)]
        p = 2 * min(np.mean([b >= 0 for b in boot]), np.mean([b <= 0 for b in boot]))
        return {
            "mean_delta": round(float(np.mean(deltas)), 5),
            "ci_lo": round(float(ci_lo), 5), "ci_hi": round(float(ci_hi), 5),
            "p_value": round(float(p), 4),
            "significant_95": bool(ci_lo > 0 or ci_hi < 0),
        }

    model_results = {"model": model_name, "family": family}

    for ds_name, target_text in DATASETS.items():
        print(f"\n  Dataset: {ds_name}")
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        except Exception as e:
            model_results[ds_name] = {"error": str(e)}
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = [qid for qid in list(queries.keys()) if qid in qrels]
        query_texts = [queries[q] for q in query_ids]
        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True))
        target_emb = embed_model.encode(target_text, normalize_embeddings=True)

        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)

        ds_result = {
            "n_queries": len(query_ids),
            "baseline_mean": round(float(np.mean(base_pq)), 5),
        }

        # Strategy 1: Addition uniform
        q_add = normalize_rows(query_embs + ALPHA * target_emb)
        sims_add = q_add @ corpus_embs.T
        add_pq = per_query_ndcg(query_ids, doc_ids, sims_add, qrels)
        ds_result["addition_uniform"] = bootstrap_ci(base_pq, add_pq)
        print(f"    Addition: Δ={ds_result['addition_uniform']['mean_delta']:+.5f} p={ds_result['addition_uniform']['p_value']:.3f}")

        # Strategy 2: Adaptive alpha
        adapt_scores = np.zeros_like(sims_base)
        for i in range(len(query_ids)):
            sim = float(query_embs[i] @ target_emb)
            alpha_q = ALPHA * (1 - sim) ** 2
            q_rot = normalize_vec(query_embs[i] + alpha_q * target_emb)
            adapt_scores[i] = q_rot @ corpus_embs.T
        adapt_pq = per_query_ndcg(query_ids, doc_ids, adapt_scores, qrels)
        ds_result["adaptive_alpha"] = bootstrap_ci(base_pq, adapt_pq)
        print(f"    Adaptive: Δ={ds_result['adaptive_alpha']['mean_delta']:+.5f} p={ds_result['adaptive_alpha']['p_value']:.3f}")

        # Strategy 3: Multi-vector RRF (generic)
        mv_scores = np.zeros((len(query_ids), len(doc_ids)))
        for i in range(len(query_ids)):
            rrf = rrf_fusion([sims_base[i], sims_add[i]], k=RRF_K)
            mv_scores[i] = rrf
        mv_pq = per_query_ndcg(query_ids, doc_ids, mv_scores, qrels)
        ds_result["multivector_generic"] = bootstrap_ci(base_pq, mv_pq)
        print(f"    MV_gen:   Δ={ds_result['multivector_generic']['mean_delta']:+.5f} p={ds_result['multivector_generic']['p_value']:.3f}")

        # Strategy 4: Full stack (top-k + adaptive + MV)
        corpus_tk, comps = remove_top_k(corpus_embs, k=1)
        query_tk = query_embs.copy()
        for c in comps:
            query_tk -= (query_tk @ c).reshape(-1, 1) * c
        query_tk = normalize_rows(query_tk)
        target_tk = target_emb.copy()
        for c in comps:
            target_tk -= np.dot(target_tk, c) * c
        target_tk = normalize_vec(target_tk)

        sims_tk = query_tk @ corpus_tk.T
        full_scores = np.zeros((len(query_ids), len(doc_ids)))
        for i in range(len(query_ids)):
            sim = float(query_tk[i] @ target_tk)
            alpha_q = ALPHA * (1 - sim) ** 2
            q_rot = normalize_vec(query_tk[i] + alpha_q * target_tk)
            sims_rot = q_rot @ corpus_tk.T
            full_scores[i] = rrf_fusion([sims_tk[i], sims_rot], k=RRF_K)
        full_pq = per_query_ndcg(query_ids, doc_ids, full_scores, qrels)
        ds_result["full_stack"] = bootstrap_ci(base_pq, full_pq)
        print(f"    Full:     Δ={ds_result['full_stack']['mean_delta']:+.5f} p={ds_result['full_stack']['p_value']:.3f}")

        # Strategy 5: Adaptive stack (adaptive + MV, NO top-k)
        adapt_mv_scores = np.zeros((len(query_ids), len(doc_ids)))
        for i in range(len(query_ids)):
            sim = float(query_embs[i] @ target_emb)
            alpha_q = ALPHA * (1 - sim) ** 2
            q_rot = normalize_vec(query_embs[i] + alpha_q * target_emb)
            sims_rot = q_rot @ corpus_embs.T
            adapt_mv_scores[i] = rrf_fusion([sims_base[i], sims_rot], k=RRF_K)
        adapt_mv_pq = per_query_ndcg(query_ids, doc_ids, adapt_mv_scores, qrels)
        ds_result["adaptive_stack_no_topk"] = bootstrap_ci(base_pq, adapt_mv_pq)
        print(f"    AdptStack:Δ={ds_result['adaptive_stack_no_topk']['mean_delta']:+.5f} p={ds_result['adaptive_stack_no_topk']['p_value']:.3f}")

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/significance_full/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Full Significance (6 models x 5 datasets x 5 strategies)")
    print("=" * 70)
    results = list(run_significance.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
