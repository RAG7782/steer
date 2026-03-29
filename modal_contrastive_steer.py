"""
STEER — Contrastive Steer Operation

Validates: normalize(q + alpha*t_pos - beta*t_neg)
Simultaneously push toward positive target and away from negative.
Tests symmetric (alpha=beta) and asymmetric grids.

6 models x 5 datasets. Bootstrap 1000x integrated.

Usage: modal run modal_contrastive_steer.py
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

app = modal.App("steer-contrastive", image=image)
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
        "target_positive": "clinical medicine and patient outcomes",
        "target_negative": "animal model laboratory studies",
    },
    "arguana": {
        "target_positive": "legal reasoning and jurisprudence",
        "target_negative": "informal opinion and anecdote",
    },
    "nfcorpus": {
        "target_positive": "clinical nutrition interventions",
        "target_negative": "food industry marketing",
    },
    "fiqa": {
        "target_positive": "macroeconomic policy impacts",
        "target_negative": "personal finance anecdotes",
    },
    "trec-covid": {
        "target_positive": "COVID-19 clinical treatment protocols",
        "target_negative": "COVID-19 misinformation and conspiracy",
    },
}

SYMMETRIC_ALPHAS = [0.05, 0.1, 0.2, 0.3]
ASYM_GRID = [(0.1, 0.05), (0.1, 0.1), (0.1, 0.2), (0.2, 0.05), (0.2, 0.1), (0.2, 0.2)]
MAX_QUERIES = 100
N_BOOTSTRAP = 1000


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_contrastive(model_name: str, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Contrastive Steer: {model_name} ({family})")
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
        pos_emb = embed_model.encode(ds_cfg["target_positive"], normalize_embeddings=True)
        neg_emb = embed_model.encode(ds_cfg["target_negative"], normalize_embeddings=True)

        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)

        ds_result = {
            "baseline_ndcg10": round(float(np.mean(base_pq)), 5),
            "n_queries": len(query_ids),
            "pos_neg_similarity": round(float(pos_emb @ neg_emb), 4),
            "symmetric": {},
            "asymmetric": {},
            "positive_only": {},
        }

        # Positive-only baseline for comparison
        for alpha in SYMMETRIC_ALPHAS:
            q_pos = normalize_rows(query_embs + alpha * pos_emb)
            pos_pq = per_query_ndcg(query_ids, doc_ids, q_pos @ corpus_embs.T, qrels)
            ds_result["positive_only"][f"alpha_{alpha}"] = bootstrap_ci(base_pq, pos_pq)

        # Symmetric contrastive
        for alpha in SYMMETRIC_ALPHAS:
            q_contr = normalize_rows(query_embs + alpha * pos_emb - alpha * neg_emb)
            contr_pq = per_query_ndcg(query_ids, doc_ids, q_contr @ corpus_embs.T, qrels)
            boot = bootstrap_ci(base_pq, contr_pq)
            pos_delta = ds_result["positive_only"][f"alpha_{alpha}"]["mean_delta"]
            ds_result["symmetric"][f"alpha_{alpha}"] = {
                **boot,
                "improvement_over_positive_only": round(boot["mean_delta"] - pos_delta, 5),
            }
            print(f"    Sym α={alpha}: Δ={boot['mean_delta']:+.5f} (vs pos_only: {boot['mean_delta']-pos_delta:+.5f})")

        # Asymmetric contrastive
        for alpha_pos, beta_neg in ASYM_GRID:
            q_asym = normalize_rows(query_embs + alpha_pos * pos_emb - beta_neg * neg_emb)
            asym_pq = per_query_ndcg(query_ids, doc_ids, q_asym @ corpus_embs.T, qrels)
            boot = bootstrap_ci(base_pq, asym_pq)
            ds_result["asymmetric"][f"a{alpha_pos}_b{beta_neg}"] = boot
            print(f"    Asym α+={alpha_pos} β-={beta_neg}: Δ={boot['mean_delta']:+.5f}")

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/contrastive_steer/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Contrastive Steer (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_contrastive.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
