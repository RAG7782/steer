"""
A2RAG — Statistical Significance via Bootstrap

Valida que os deltas observados nos experimentos sao estatisticamente significativos.
Bootstrap confidence intervals (95%) para cada estrategia vs baseline.

Metodo: para cada estrategia, resample queries com reposicao 1000x,
computar nDCG@10 em cada resample, reportar CI e p-value.

Testa as estrategias mais importantes:
1. Addition uniforme (referencia negativa)
2. Multi-vector per-query targets
3. Full stack (top-k + adaptive + MV per-query)
4. Adaptive alpha only
5. Multi-vector generic (RRF)

6 modelos x 2 datasets x 5 estrategias.
Nao precisa de LLM — usa os embeddings diretamente.

Usage: modal run modal_significance.py

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
        "scikit-learn",
    )
)

app = modal.App("a2rag-significance", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("all-mpnet-base-v2", "trained-1B-pairs"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("intfloat/e5-small-v2", "instruction-tuned"),
    ("thenlper/gte-small", "general"),
]

DATASETS = ["scifact", "arguana"]

ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}

ALPHA_MAX = 0.2
RRF_K = 60
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_significance(model_name: str, family: str):
    """Bootstrap significance test for one model."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Significance: {model_name} ({family})")
    print(f"{'='*60}")

    model = SentenceTransformer(model_name)
    model_results = {"model": model_name, "family": family}

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def compute_per_query_ndcg10(sims, query_ids, doc_ids, qrels):
        """Compute nDCG@10 per query. Returns array of per-query scores."""
        per_query = []
        for i, qid in enumerate(query_ids):
            if qid not in qrels:
                continue
            top = np.argsort(sims[i])[::-1][:10]
            dcg = 0
            for rank, idx in enumerate(top):
                rel = qrels.get(qid, {}).get(doc_ids[idx], 0)
                dcg += rel / np.log2(rank + 2)
            ideal_rels = sorted(qrels.get(qid, {}).values(), reverse=True)[:10]
            idcg = sum(r / np.log2(j + 2) for j, r in enumerate(ideal_rels))
            per_query.append(dcg / idcg if idcg > 0 else 0)
        return np.array(per_query)

    def compute_per_query_ndcg10_from_results(results_dict, query_ids, qrels):
        """Compute nDCG@10 per query from results dict."""
        per_query = []
        for qid in query_ids:
            if qid not in qrels or qid not in results_dict:
                continue
            sorted_docs = sorted(results_dict[qid].items(), key=lambda x: x[1], reverse=True)[:10]
            dcg = 0
            for rank, (did, _) in enumerate(sorted_docs):
                rel = qrels.get(qid, {}).get(did, 0)
                dcg += rel / np.log2(rank + 2)
            ideal_rels = sorted(qrels.get(qid, {}).values(), reverse=True)[:10]
            idcg = sum(r / np.log2(j + 2) for j, r in enumerate(ideal_rels))
            per_query.append(dcg / idcg if idcg > 0 else 0)
        return np.array(per_query)

    def bootstrap_ci(baseline_scores, treatment_scores, n_boot=N_BOOTSTRAP, ci=0.95):
        """Bootstrap CI for mean(treatment - baseline)."""
        np.random.seed(42)
        n = len(baseline_scores)
        deltas = treatment_scores - baseline_scores
        boot_means = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            boot_means.append(np.mean(deltas[idx]))
        boot_means = np.array(sorted(boot_means))
        alpha = (1 - ci) / 2
        lo = boot_means[int(alpha * n_boot)]
        hi = boot_means[int((1 - alpha) * n_boot)]
        mean_delta = np.mean(deltas)
        # p-value: proportion of bootstrap samples where delta <= 0
        p_value = np.mean(boot_means <= 0) if mean_delta > 0 else np.mean(boot_means >= 0)
        return {
            "mean_delta": round(float(mean_delta), 5),
            "ci_lo": round(float(lo), 5),
            "ci_hi": round(float(hi), 5),
            "p_value": round(float(p_value), 4),
            "significant_95": bool(lo > 0 or hi < 0),
        }

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = [qid for qid in queries.keys() if qid in qrels]
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        # Top-k removal
        pca = PCA()
        pca.fit(corpus_embs)
        corpus_tk = corpus_embs.copy()
        query_tk = query_embs.copy()
        comp = pca.components_[0]
        corpus_tk -= (corpus_tk @ comp).reshape(-1, 1) * comp
        query_tk -= (query_tk @ comp).reshape(-1, 1) * comp
        corpus_tk = normalize_rows(corpus_tk)
        query_tk = normalize_rows(query_tk)
        target_tk = target_emb - np.dot(target_emb, comp) * comp
        target_tk = normalize_vec(target_tk)

        # === Baseline per-query nDCG@10 ===
        sims_base = query_embs @ corpus_embs.T
        base_pq = compute_per_query_ndcg10(sims_base, query_ids, doc_ids, qrels)
        print(f"    Baseline: mean={np.mean(base_pq):.4f}, n_queries={len(base_pq)}")

        ds_results = {"n_queries": len(base_pq), "baseline_mean": round(float(np.mean(base_pq)), 4)}

        # === 1. Addition uniforme ===
        q_add = normalize_rows(query_embs + ALPHA_MAX * target_emb)
        sims_add = q_add @ corpus_embs.T
        add_pq = compute_per_query_ndcg10(sims_add, query_ids, doc_ids, qrels)
        ds_results["addition_uniform"] = bootstrap_ci(base_pq, add_pq)
        print(f"    Addition uniform: {ds_results['addition_uniform']}")

        # === 2. Adaptive alpha (quadratic) ===
        adapt_scores = np.zeros_like(sims_base)
        for i in range(len(query_ids)):
            sim = float(query_embs[i] @ target_emb)
            alpha = ALPHA_MAX * (1 - sim) ** 2
            q_rot = normalize_vec(query_embs[i] + alpha * target_emb)
            adapt_scores[i] = q_rot @ corpus_embs.T
        adapt_pq = compute_per_query_ndcg10(adapt_scores, query_ids, doc_ids, qrels)
        ds_results["adaptive_alpha"] = bootstrap_ci(base_pq, adapt_pq)
        print(f"    Adaptive alpha: {ds_results['adaptive_alpha']}")

        # === 3. Multi-vector RRF (generic target) ===
        mv_results = {}
        for i, qid in enumerate(query_ids):
            q_rot = normalize_vec(query_embs[i] + ALPHA_MAX * target_emb)
            sims_rot = q_rot @ corpus_embs.T
            ranks_base = np.argsort(np.argsort(-sims_base[i])) + 1
            ranks_rot = np.argsort(np.argsort(-sims_rot)) + 1
            rrf = 1.0 / (RRF_K + ranks_base) + 1.0 / (RRF_K + ranks_rot)
            top = np.argsort(rrf)[::-1][:100]
            mv_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        mv_pq = compute_per_query_ndcg10_from_results(mv_results, query_ids, qrels)
        ds_results["multivector_generic"] = bootstrap_ci(base_pq, mv_pq)
        print(f"    Multi-vector generic: {ds_results['multivector_generic']}")

        # === 4. Full stack (top-k + adaptive + MV) ===
        sims_base_tk = query_tk @ corpus_tk.T
        stack_results = {}
        for i, qid in enumerate(query_ids):
            sim = float(query_tk[i] @ target_tk)
            alpha = ALPHA_MAX * (1 - sim) ** 2
            q_rot = normalize_vec(query_tk[i] + alpha * target_tk)
            sims_rot = q_rot @ corpus_tk.T
            ranks_base = np.argsort(np.argsort(-sims_base_tk[i])) + 1
            ranks_rot = np.argsort(np.argsort(-sims_rot)) + 1
            rrf = 1.0 / (RRF_K + ranks_base) + 1.0 / (RRF_K + ranks_rot)
            top = np.argsort(rrf)[::-1][:100]
            stack_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        stack_pq = compute_per_query_ndcg10_from_results(stack_results, query_ids, qrels)
        ds_results["full_stack"] = bootstrap_ci(base_pq, stack_pq)
        print(f"    Full stack: {ds_results['full_stack']}")

        model_results[ds_name] = ds_results

    safe = model_name.replace("/", "_")
    out_path = f"/results/significance/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Statistical Significance (Bootstrap)")
    print("  6 models x 2 datasets x 4 strategies x 1000 bootstraps")
    print("=" * 70)

    all_results = list(run_significance.starmap(
        [(name, fam) for name, fam in MODELS]
    ))

    print("\n" + "=" * 70)
    print("  SUMMARY: Significant results (p < 0.05)")
    print("=" * 70)

    for r in all_results:
        for ds in DATASETS:
            if ds not in r: continue
            d = r[ds]
            for strat in ["addition_uniform", "adaptive_alpha", "multivector_generic", "full_stack"]:
                if strat not in d: continue
                s = d[strat]
                sig = "***" if s["p_value"] < 0.01 else "**" if s["p_value"] < 0.05 else ""
                if sig:
                    print(f"  {r['model']:<30} {ds:<10} {strat:<22} Δ={s['mean_delta']:+.5f} CI=[{s['ci_lo']:+.5f}, {s['ci_hi']:+.5f}] p={s['p_value']:.4f} {sig}")

    print("\n  DONE!")
