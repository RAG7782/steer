"""
STEER — Orbit + Gradient Walk Operations

Validates:
- Orbit: N targets generate N separate result sets. Measures diversity.
- Gradient Walk: alpha from 0 to 0.5 in fine steps. Produces degradation curve.

6 models x 5 datasets. No bootstrap (curves/sets ARE the result).

Usage: modal run modal_orbit_gradient_walk.py
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

app = modal.App("steer-orbit-gradient", image=image)
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
        "orbit_targets": [
            "pharmacological interventions",
            "epidemiological surveillance",
            "genetic mechanisms and pathways",
            "clinical trial design",
            "public health policy",
        ],
    },
    "arguana": {
        "target": "legal reasoning and jurisprudence",
        "orbit_targets": [
            "economic policy arguments",
            "ethical philosophy reasoning",
            "historical precedent analysis",
            "scientific evidence interpretation",
            "political ideology critique",
        ],
    },
    "nfcorpus": {
        "target": "clinical nutrition interventions",
        "orbit_targets": [
            "pharmaceutical nutrition supplements",
            "metabolic disease pathways",
            "public health dietary guidelines",
            "food chemistry and processing",
            "microbiome and gut health",
        ],
    },
    "fiqa": {
        "target": "macroeconomic policy impacts",
        "orbit_targets": [
            "regulatory compliance frameworks",
            "behavioral economics insights",
            "quantitative risk modeling",
            "corporate governance practices",
            "global trade dynamics",
        ],
    },
    "trec-covid": {
        "target": "COVID-19 clinical treatment protocols",
        "orbit_targets": [
            "vaccine development and immunology",
            "epidemiological modeling and forecasting",
            "healthcare system capacity planning",
            "mental health during pandemic",
            "economic impact of lockdowns",
        ],
    },
}

GRADIENT_ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
ORBIT_ALPHA = 0.1
MAX_QUERIES = 100


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_orbit_gradient(model_name: str, family: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Orbit + Gradient Walk: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

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
        target_emb = embed_model.encode(ds_cfg["target"], normalize_embeddings=True)

        # Baseline
        sims_base = query_embs @ corpus_embs.T
        base_pq = per_query_ndcg(query_ids, doc_ids, sims_base, qrels)
        baseline_ndcg = float(np.mean(base_pq))

        # === GRADIENT WALK ===
        print(f"    Gradient Walk...")
        curve = []
        for alpha in GRADIENT_ALPHAS:
            if alpha == 0:
                ndcg = baseline_ndcg
            else:
                q_steered = normalize_rows(query_embs + alpha * target_emb)
                sims_steered = q_steered @ corpus_embs.T
                steered_pq = per_query_ndcg(query_ids, doc_ids, sims_steered, qrels)
                ndcg = float(np.mean(steered_pq))
            pct = ndcg / max(baseline_ndcg, 1e-10)
            curve.append({"alpha": alpha, "ndcg10": round(ndcg, 5), "pct_baseline": round(pct, 5)})
            print(f"      α={alpha:.3f}: nDCG={ndcg:.5f} ({pct:.3%})")

        # Find inflection points
        inflection_95, inflection_90 = None, None
        for pt in curve:
            if inflection_95 is None and pt["pct_baseline"] < 0.95:
                inflection_95 = pt["alpha"]
            if inflection_90 is None and pt["pct_baseline"] < 0.90:
                inflection_90 = pt["alpha"]

        # Check monotonicity
        ndcg_vals = [pt["ndcg10"] for pt in curve]
        monotonic = all(ndcg_vals[i] >= ndcg_vals[i+1] - 0.001 for i in range(len(ndcg_vals)-1))

        # === ORBIT ===
        print(f"    Orbit (5 arms, α={ORBIT_ALPHA})...")
        orbit_targets = ds_cfg["orbit_targets"]
        orbit_embs = np.array(embed_model.encode(orbit_targets, normalize_embeddings=True))

        orbit_results = []
        orbit_top_sets = []  # list of list of sets

        for arm_idx, (t_text, t_emb) in enumerate(zip(orbit_targets, orbit_embs)):
            q_orbit = normalize_rows(query_embs + ORBIT_ALPHA * t_emb)
            sims_orbit = q_orbit @ corpus_embs.T
            orbit_pq = per_query_ndcg(query_ids, doc_ids, sims_orbit, qrels)

            # Top-10 per query for this arm
            arm_tops = [set(np.argsort(sims_orbit[i])[::-1][:10]) for i in range(len(query_ids))]
            orbit_top_sets.append(arm_tops)

            orbit_results.append({
                "target": t_text,
                "ndcg10": round(float(np.mean(orbit_pq)), 5),
                "delta": round(float(np.mean(orbit_pq)) - baseline_ndcg, 5),
            })
            print(f"      Arm {arm_idx}: {t_text[:30]:30s} Δ={float(np.mean(orbit_pq))-baseline_ndcg:+.5f}")

        # Baseline top-10 per query
        base_tops = [set(np.argsort(sims_base[i])[::-1][:10]) for i in range(len(query_ids))]

        # Pairwise Jaccard between arms
        n_arms = len(orbit_top_sets)
        pairwise_jaccard = []
        for a in range(n_arms):
            for b in range(a+1, n_arms):
                j_vals = []
                for qi in range(len(query_ids)):
                    inter = len(orbit_top_sets[a][qi] & orbit_top_sets[b][qi])
                    union = len(orbit_top_sets[a][qi] | orbit_top_sets[b][qi])
                    j_vals.append(inter / max(union, 1))
                pairwise_jaccard.append(float(np.mean(j_vals)))
        mean_pairwise_jaccard = float(np.mean(pairwise_jaccard))

        # Coverage: union of all arms vs baseline
        unique_docs_per_query = []
        for qi in range(len(query_ids)):
            union_arms = set()
            for arm in orbit_top_sets:
                union_arms |= arm[qi]
            new_docs = union_arms - base_tops[qi]
            unique_docs_per_query.append(len(new_docs))
        mean_new_docs = float(np.mean(unique_docs_per_query))

        ds_result = {
            "baseline_ndcg10": round(baseline_ndcg, 5),
            "n_queries": len(query_ids),
            "gradient_walk": {
                "curve": curve,
                "inflection_95": inflection_95,
                "inflection_90": inflection_90,
                "monotonic": monotonic,
            },
            "orbit": {
                "alpha": ORBIT_ALPHA,
                "arms": orbit_results,
                "mean_pairwise_jaccard": round(mean_pairwise_jaccard, 4),
                "mean_new_docs_vs_baseline": round(mean_new_docs, 2),
                "diversity_achieved": mean_pairwise_jaccard < 0.5,
            },
        }

        model_results[ds_name] = ds_result

    safe = model_name.replace("/", "_")
    out_path = f"/results/orbit_gradient_walk/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Orbit + Gradient Walk (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_orbit_gradient.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
