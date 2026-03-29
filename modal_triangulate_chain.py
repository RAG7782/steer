"""
STEER — Triangulate + Steer Chain Operations

Validates:
- Triangulate: steer(steer(q, t1, a1), t2, a2) — sequential 2-target steering
- Steer Chain: N sequential steerings. Tests if composition is linear.

6 models x 5 datasets. Bootstrap 1000x for triangulate.

Usage: modal run modal_triangulate_chain.py
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

app = modal.App("steer-triangulate-chain", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("all-mpnet-base-v2", "trained"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("intfloat/e5-small-v2", "instruction"),
    ("thenlper/gte-small", "general"),
]

DATASETS_CONFIG = {
    "scifact": {
        "target1": "clinical medicine and patient outcomes",
        "target2": "genetic mechanisms and pathways",
    },
    "arguana": {
        "target1": "legal reasoning and jurisprudence",
        "target2": "ethical philosophy reasoning",
    },
    "nfcorpus": {
        "target1": "clinical nutrition interventions",
        "target2": "metabolic disease pathways",
    },
    "fiqa": {
        "target1": "macroeconomic policy impacts",
        "target2": "regulatory compliance frameworks",
    },
    "trec-covid": {
        "target1": "COVID-19 clinical treatment protocols",
        "target2": "vaccine development and immunology",
    },
}

TRI_GRID = [(0.05, 0.05), (0.05, 0.1), (0.1, 0.05), (0.1, 0.1), (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)]
CHAIN_NS = [1, 2, 3, 5, 10]
CHAIN_ALPHA = 0.1
RRF_K = 60
MAX_QUERIES = 100
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_triangulate_chain(model_name: str, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Triangulate + Chain: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

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

    def rrf_fusion(scores_list, k=60):
        n = scores_list[0].shape[0]
        rrf = np.zeros(n)
        for s in scores_list:
            ranks = np.argsort(np.argsort(-s)) + 1
            rrf += 1.0 / (k + ranks)
        return rrf

    model_results = {"model": model_name, "family": family}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
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
        query_ids = [qid for qid in list(queries.keys())[:MAX_QUERIES] if qid in qrels]
        query_texts = [queries[q] for q in query_ids]
        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True))
        t1_emb = embed_model.encode(ds_cfg["target1"], normalize_embeddings=True)
        t2_emb = embed_model.encode(ds_cfg["target2"], normalize_embeddings=True)

        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)
        baseline_ndcg = float(np.mean(base_pq))

        # Consensus baseline: steer to mean(t1, t2)
        consensus_t = normalize_vec((t1_emb + t2_emb) / 2)

        ds_result = {
            "baseline_ndcg10": round(baseline_ndcg, 5),
            "n_queries": len(query_ids),
            "t1_t2_similarity": round(float(t1_emb @ t2_emb), 4),
            "triangulate": {},
            "consensus_comparison": {},
            "multiview_comparison": {},
            "chain": {},
        }

        # === TRIANGULATE ===
        best_tri_delta = -999
        for a1, a2 in TRI_GRID:
            # Step 1: steer toward t1
            q_step1 = normalize_rows(query_embs + a1 * t1_emb)
            # Step 2: steer result toward t2
            q_step2 = normalize_rows(q_step1 + a2 * t2_emb)
            sims_tri = q_step2 @ corpus_embs.T
            tri_pq = per_query_ndcg(query_ids, doc_ids, sims_tri, qrels)
            boot = bootstrap_ci(base_pq, tri_pq)
            ds_result["triangulate"][f"a1_{a1}_a2_{a2}"] = boot
            if boot["mean_delta"] > best_tri_delta:
                best_tri_delta = boot["mean_delta"]
            print(f"    Tri α1={a1} α2={a2}: Δ={boot['mean_delta']:+.5f}")

        # Consensus comparison at equivalent total alpha
        for total_alpha in [0.1, 0.2, 0.3]:
            q_cons = normalize_rows(query_embs + total_alpha * consensus_t)
            cons_pq = per_query_ndcg(query_ids, doc_ids, q_cons @ corpus_embs.T, qrels)
            boot = bootstrap_ci(base_pq, cons_pq)
            ds_result["consensus_comparison"][f"alpha_{total_alpha}"] = boot

        # Multi-view comparison: search t1 and t2 separately, RRF
        for alpha in [0.1, 0.2]:
            q_t1 = normalize_rows(query_embs + alpha * t1_emb)
            q_t2 = normalize_rows(query_embs + alpha * t2_emb)
            mv_scores = np.zeros_like(sims_base)
            for i in range(len(query_ids)):
                mv_scores[i] = rrf_fusion([sims_base[i], (q_t1 @ corpus_embs.T)[i], (q_t2 @ corpus_embs.T)[i]], k=RRF_K)
            mv_pq = per_query_ndcg(query_ids, doc_ids, mv_scores, qrels)
            boot = bootstrap_ci(base_pq, mv_pq)
            ds_result["multiview_comparison"][f"alpha_{alpha}"] = boot
            print(f"    MV α={alpha}: Δ={boot['mean_delta']:+.5f}")

        # === STEER CHAIN ===
        for n_steps in CHAIN_NS:
            q_chain = query_embs.copy()
            for _ in range(n_steps):
                q_chain = normalize_rows(q_chain + CHAIN_ALPHA * t1_emb)
            sims_chain = q_chain @ corpus_embs.T
            chain_pq = per_query_ndcg(query_ids, doc_ids, sims_chain, qrels)

            # Compare with single-step equivalent: steer(q, t, n*alpha)
            equiv_alpha = n_steps * CHAIN_ALPHA
            q_single = normalize_rows(query_embs + equiv_alpha * t1_emb)
            single_pq = per_query_ndcg(query_ids, doc_ids, q_single @ corpus_embs.T, qrels)

            ds_result["chain"][f"N_{n_steps}"] = {
                "ndcg10": round(float(np.mean(chain_pq)), 5),
                "delta": round(float(np.mean(chain_pq)) - baseline_ndcg, 5),
                "equiv_single_step_alpha": equiv_alpha,
                "equiv_single_ndcg10": round(float(np.mean(single_pq)), 5),
                "equiv_single_delta": round(float(np.mean(single_pq)) - baseline_ndcg, 5),
                "chain_vs_single_diff": round(float(np.mean(chain_pq) - np.mean(single_pq)), 5),
            }
            print(f"    Chain N={n_steps}: Δ={float(np.mean(chain_pq))-baseline_ndcg:+.5f} (single α={equiv_alpha}: {float(np.mean(single_pq))-baseline_ndcg:+.5f})")

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/triangulate_chain/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Triangulate + Chain (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_triangulate_chain.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
