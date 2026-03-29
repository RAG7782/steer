"""
STEER — Centroid Operations: Amplify, Diffuse, Consensus

Validates:
- Amplify: steer(q, centroid, +alpha) — intensify specificity
- Diffuse: steer(q, centroid, -alpha) — generalize, increase diversity
- Consensus: steer(q, mean([t1..tn]), alpha) — average of multiple targets

6 models x 5 datasets x 4 alphas. Bootstrap 1000x integrated.

Usage: modal run modal_centroid_ops.py
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

app = modal.App("steer-centroid-ops", image=image)
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
        "orbit_targets": [
            "pharmacological interventions",
            "epidemiological surveillance",
            "genetic mechanisms and pathways",
            "clinical trial design",
            "public health policy",
        ],
    },
    "arguana": {
        "orbit_targets": [
            "economic policy arguments",
            "ethical philosophy reasoning",
            "historical precedent analysis",
            "scientific evidence interpretation",
            "political ideology critique",
        ],
    },
    "nfcorpus": {
        "orbit_targets": [
            "pharmaceutical nutrition supplements",
            "metabolic disease pathways",
            "public health dietary guidelines",
            "food chemistry and processing",
            "microbiome and gut health",
        ],
    },
    "fiqa": {
        "orbit_targets": [
            "regulatory compliance frameworks",
            "behavioral economics insights",
            "quantitative risk modeling",
            "corporate governance practices",
            "global trade dynamics",
        ],
    },
    "trec-covid": {
        "orbit_targets": [
            "vaccine development and immunology",
            "epidemiological modeling and forecasting",
            "healthcare system capacity planning",
            "mental health during pandemic",
            "economic impact of lockdowns",
        ],
    },
}

ALPHAS = [0.05, 0.1, 0.2, 0.3]
MAX_QUERIES = 100
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_centroid_ops(model_name: str, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Centroid Ops: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

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
            dcg = sum(qrels_dict[qid].get(doc_ids[idx], 0) / np.log2(rank + 2) for rank, idx in enumerate(top_idx))
            rels = sorted(qrels_dict[qid].values(), reverse=True)[:k]
            idcg = sum(r / np.log2(j + 2) for j, r in enumerate(rels))
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

    def jaccard_at_k(scores_a, scores_b, k=10):
        n = scores_a.shape[0]
        return float(np.mean([
            len(set(np.argsort(scores_a[i])[::-1][:k]) & set(np.argsort(scores_b[i])[::-1][:k])) /
            len(set(np.argsort(scores_a[i])[::-1][:k]) | set(np.argsort(scores_b[i])[::-1][:k]))
            for i in range(n)
        ]))

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

        # Compute corpus centroid
        centroid = normalize_vec(np.mean(corpus_embs, axis=0))

        # Compute isotropy metric
        mean_cos = float(np.mean(corpus_embs @ centroid))
        print(f"    Centroid mean_cos={mean_cos:.4f} (isotropy proxy)")

        # Encode orbit targets for consensus
        orbit_texts = ds_cfg["orbit_targets"]
        orbit_embs = np.array(embed_model.encode(orbit_texts, normalize_embeddings=True))
        consensus_target = normalize_vec(np.mean(orbit_embs, axis=0))

        # Baseline
        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)

        ds_result = {
            "baseline_ndcg10": round(float(np.mean(base_pq)), 4),
            "n_queries": len(query_ids),
            "centroid_mean_cos": round(mean_cos, 4),
            "amplify": {},
            "diffuse": {},
            "consensus": {},
        }

        for alpha in ALPHAS:
            # === AMPLIFY: toward centroid ===
            q_amp = normalize_rows(query_embs + alpha * centroid)
            sims_amp = q_amp @ corpus_embs.T
            amp_pq = per_query_ndcg(query_ids, doc_ids, sims_amp, qrels)
            boot_amp = bootstrap_ci(base_pq, amp_pq)
            jaccard_amp = jaccard_at_k(sims_base, sims_amp)
            ds_result["amplify"][f"alpha_{alpha}"] = {"jaccard_at_10": round(jaccard_amp, 4), **boot_amp}
            print(f"    Amplify α={alpha}: Δ={boot_amp['mean_delta']:+.5f} J={jaccard_amp:.3f}")

            # === DIFFUSE: away from centroid ===
            q_diff = normalize_rows(query_embs - alpha * centroid)
            sims_diff = q_diff @ corpus_embs.T
            diff_pq = per_query_ndcg(query_ids, doc_ids, sims_diff, qrels)
            boot_diff = bootstrap_ci(base_pq, diff_pq)
            jaccard_diff = jaccard_at_k(sims_base, sims_diff)
            ds_result["diffuse"][f"alpha_{alpha}"] = {"jaccard_at_10": round(jaccard_diff, 4), **boot_diff}
            print(f"    Diffuse α={alpha}: Δ={boot_diff['mean_delta']:+.5f} J={jaccard_diff:.3f}")

            # === CONSENSUS: mean of orbit targets ===
            q_cons = normalize_rows(query_embs + alpha * consensus_target)
            sims_cons = q_cons @ corpus_embs.T
            cons_pq = per_query_ndcg(query_ids, doc_ids, sims_cons, qrels)
            boot_cons = bootstrap_ci(base_pq, cons_pq)

            # Also test individual orbit targets for variance comparison
            individual_deltas = []
            for t_emb in orbit_embs:
                q_ind = normalize_rows(query_embs + alpha * t_emb)
                sims_ind = q_ind @ corpus_embs.T
                ind_pq = per_query_ndcg(query_ids, doc_ids, sims_ind, qrels)
                individual_deltas.append(float(np.mean(ind_pq) - np.mean(base_pq)))

            ds_result["consensus"][f"alpha_{alpha}"] = {
                **boot_cons,
                "individual_target_deltas": [round(d, 5) for d in individual_deltas],
                "individual_variance": round(float(np.var(individual_deltas)), 8),
                "consensus_vs_mean_individual": round(boot_cons["mean_delta"] - float(np.mean(individual_deltas)), 5),
            }
            print(f"    Consensus α={alpha}: Δ={boot_cons['mean_delta']:+.5f} ind_var={np.var(individual_deltas):.6f}")

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/centroid_ops/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Centroid Ops: Amplify + Diffuse + Consensus")
    print("=" * 70)
    results = list(run_centroid_ops.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
