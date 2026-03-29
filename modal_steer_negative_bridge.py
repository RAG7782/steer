"""
STEER — Rotate Away + Bridge Operations

Validates:
- Rotate Away: steer(q, t, -alpha) — semantic filtering
- Bridge: normalize(q + alpha*(target_emb - source_emb)) — domain transfer

6 models x 5 datasets x 4 alphas. Bootstrap 1000x integrated.

Usage: modal run modal_steer_negative_bridge.py
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

app = modal.App("steer-negative-bridge", image=image)
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
        "target": "clinical medicine and patient outcomes",
        "source_domain": "biomedical research methodology",
        "target_domain": "clinical practice guidelines",
    },
    "arguana": {
        "target": "legal reasoning and jurisprudence",
        "source_domain": "debate and argumentation",
        "target_domain": "legal precedent and case law",
    },
    "nfcorpus": {
        "target": "clinical nutrition interventions",
        "source_domain": "nutrition science research",
        "target_domain": "clinical dietary recommendations",
    },
    "fiqa": {
        "target": "macroeconomic policy impacts",
        "source_domain": "financial question answering",
        "target_domain": "institutional investment strategy",
    },
    "trec-covid": {
        "target": "COVID-19 clinical treatment protocols",
        "source_domain": "COVID-19 research papers",
        "target_domain": "pandemic public health response",
    },
}

ALPHAS = [0.05, 0.1, 0.2, 0.3]
MAX_QUERIES = 100
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_negative_bridge(model_name: str, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Rotate Away + Bridge: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def per_query_ndcg(query_ids, doc_ids, scores_matrix, qrels_dict, k=10):
        """Compute nDCG@k per query for bootstrap."""
        per_q = []
        for i, qid in enumerate(query_ids):
            if qid not in qrels_dict:
                continue
            top_idx = np.argsort(scores_matrix[i])[::-1][:k]
            dcg = 0.0
            for rank, idx in enumerate(top_idx):
                did = doc_ids[idx]
                rel = qrels_dict[qid].get(did, 0)
                dcg += rel / np.log2(rank + 2)
            # Ideal DCG
            rels = sorted(qrels_dict[qid].values(), reverse=True)[:k]
            idcg = sum(r / np.log2(j + 2) for j, r in enumerate(rels))
            per_q.append(dcg / max(idcg, 1e-10))
        return np.array(per_q)

    def bootstrap_ci(baseline_pq, method_pq, n=N_BOOTSTRAP):
        """Bootstrap confidence interval for delta."""
        rng = np.random.RandomState(42)
        deltas = method_pq - baseline_pq
        n_q = len(deltas)
        boot_means = []
        for _ in range(n):
            idx = rng.choice(n_q, n_q, replace=True)
            boot_means.append(np.mean(deltas[idx]))
        boot_means = sorted(boot_means)
        ci_lo = boot_means[int(0.025 * n)]
        ci_hi = boot_means[int(0.975 * n)]
        p = np.mean([1 if b >= 0 else 0 for b in boot_means])
        p = 2 * min(p, 1 - p)
        return {
            "mean_delta": round(float(np.mean(deltas)), 5),
            "ci_lo": round(float(ci_lo), 5),
            "ci_hi": round(float(ci_hi), 5),
            "p_value": round(float(p), 4),
            "significant_95": bool(ci_lo > 0 or ci_hi < 0),
        }

    def jaccard_at_k(scores_a, scores_b, k=10):
        """Mean Jaccard similarity between top-k of two score matrices."""
        n = scores_a.shape[0]
        jaccards = []
        for i in range(n):
            top_a = set(np.argsort(scores_a[i])[::-1][:k])
            top_b = set(np.argsort(scores_b[i])[::-1][:k])
            inter = len(top_a & top_b)
            union = len(top_a | top_b)
            jaccards.append(inter / max(union, 1))
        return float(np.mean(jaccards))

    model_results = {"model": model_name, "family": family}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
        print(f"\n  Dataset: {ds_name}")
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        except Exception as e:
            print(f"    ERROR: {e}")
            model_results[ds_name] = {"error": str(e)}
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = [qid for qid in list(queries.keys())[:MAX_QUERIES] if qid in qrels]
        query_texts = [queries[q] for q in query_ids]
        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        # Encode
        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True))
        target_emb = embed_model.encode(ds_cfg["target"], normalize_embeddings=True)
        source_emb = embed_model.encode(ds_cfg["source_domain"], normalize_embeddings=True)
        target_domain_emb = embed_model.encode(ds_cfg["target_domain"], normalize_embeddings=True)

        # Baseline
        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)

        ds_result = {
            "baseline_ndcg10": round(float(np.mean(base_pq)), 4),
            "n_queries": len(query_ids),
            "rotate_away": {},
            "bridge": {},
        }

        # === ROTATE AWAY ===
        for alpha in ALPHAS:
            q_away = normalize_rows(query_embs - alpha * target_emb)
            sims_away = q_away @ corpus_embs.T
            away_pq = per_query_ndcg(query_ids, doc_ids, sims_away, qrels)
            jaccard = jaccard_at_k(sims_base, sims_away, k=10)
            boot = bootstrap_ci(base_pq, away_pq)

            ds_result["rotate_away"][f"alpha_{alpha}"] = {
                "jaccard_at_10": round(jaccard, 4),
                **boot,
            }
            print(f"    RotAway α={alpha}: Δ={boot['mean_delta']:+.4f} Jaccard={jaccard:.3f} p={boot['p_value']:.3f}")

        # === BRIDGE ===
        for alpha in ALPHAS:
            # Bridge = steer(steer(q, source, -α), target_domain, +α)
            # Equivalent: q + α*(target_domain_emb - source_emb)
            direction = normalize_vec(target_domain_emb - source_emb)
            q_bridge = normalize_rows(query_embs + alpha * direction)
            sims_bridge = q_bridge @ corpus_embs.T
            bridge_pq = per_query_ndcg(query_ids, doc_ids, sims_bridge, qrels)
            jaccard = jaccard_at_k(sims_base, sims_bridge, k=10)
            boot = bootstrap_ci(base_pq, bridge_pq)

            # Also test two-step (should be equivalent)
            q_step1 = normalize_rows(query_embs - alpha * source_emb)
            q_step2 = normalize_rows(q_step1 + alpha * target_domain_emb)
            sims_twostep = q_step2 @ corpus_embs.T
            twostep_pq = per_query_ndcg(query_ids, doc_ids, sims_twostep, qrels)
            equivalence = float(np.mean(np.abs(bridge_pq - twostep_pq)))

            ds_result["bridge"][f"alpha_{alpha}"] = {
                "jaccard_at_10": round(jaccard, 4),
                "twostep_equivalence_error": round(equivalence, 6),
                **boot,
            }
            print(f"    Bridge α={alpha}: Δ={boot['mean_delta']:+.4f} Jaccard={jaccard:.3f} equiv_err={equivalence:.6f}")

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/steer_negative_bridge/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Rotate Away + Bridge (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_negative_bridge.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE! All models completed.")
