"""
Item 11: Projection threshold analysis.

Tests whether the 0.2 projection threshold works:
1. For each query × concept pair, compute |projection|
2. Bin queries by projection magnitude
3. Measure nDCG with/without subtraction per bin
4. If no queries have |proj| < 0.2, construct orthogonal control concepts

Hypothesis: low-projection queries should be unaffected by subtraction.

Author: Renato Aparecido Gomes
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from pytrec_eval import RelevanceEvaluator
from a2rag import subtract_orthogonal

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/item11_threshold")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana"]

SUBTRACTION_CONCEPTS = {
    "scifact": ["methodology and statistical analysis", "animal model studies",
                "genetic analysis", "clinical trials"],
    "fiqa": ["cryptocurrency", "real estate investment",
             "retirement planning", "tax optimization"],
    "nfcorpus": ["dietary supplements", "weight loss and obesity",
                 "cancer treatment", "pediatric nutrition"],
    "arguana": ["economic arguments", "moral and ethical reasoning",
                "legal precedent", "environmental impact"],
}

# Projection bins
BINS = [
    ("very_low", 0.0, 0.1),
    ("low", 0.1, 0.2),
    ("medium", 0.2, 0.3),
    ("high", 0.3, 0.5),
    ("very_high", 0.5, 1.0),
]

print(f"Model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

all_results = {}

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR / ds_name)).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    print(f"  Encoding {len(doc_texts)} docs, {len(query_texts)} queries...")
    corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                         normalize_embeddings=True, show_progress_bar=True))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                        show_progress_bar=False))

    per_query_eval = RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    # Baseline per-query
    base_sims = query_embs @ corpus_embs.T
    base_results = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(base_sims[i])[::-1][:100]
        base_results[qid] = {doc_ids[idx]: float(base_sims[i, idx]) for idx in top}
    base_pq = per_query_eval.evaluate(base_results)

    ds_results = {"dataset": ds_name, "concepts": {}}

    for concept in SUBTRACTION_CONCEPTS[ds_name]:
        concept_emb = model.encode(concept, normalize_embeddings=True)

        # Compute projections for all queries
        projections = np.abs(query_embs @ concept_emb)

        # Subtraction per-query
        sub_embs = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
        sub_sims = sub_embs @ corpus_embs.T
        sub_results = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sub_sims[i])[::-1][:100]
            sub_results[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
        sub_pq = per_query_eval.evaluate(sub_results)

        # Bin analysis
        concept_result = {
            "projection_distribution": {
                "mean": float(projections.mean()),
                "std": float(projections.std()),
                "min": float(projections.min()),
                "max": float(projections.max()),
                "percentiles": {
                    "p10": float(np.percentile(projections, 10)),
                    "p25": float(np.percentile(projections, 25)),
                    "p50": float(np.percentile(projections, 50)),
                    "p75": float(np.percentile(projections, 75)),
                    "p90": float(np.percentile(projections, 90)),
                },
            },
            "bins": {},
        }

        valid_qids = [qid for qid in query_ids if qid in base_pq and qid in sub_pq]

        print(f"\n  Concept: '{concept}'")
        print(f"    Projection: mean={projections.mean():.3f} std={projections.std():.3f} "
              f"range=[{projections.min():.3f}, {projections.max():.3f}]")

        for bin_name, lo, hi in BINS:
            # Find queries in this bin
            bin_qids = []
            for j, qid in enumerate(valid_qids):
                idx = query_ids.index(qid)
                if lo <= projections[idx] < hi:
                    bin_qids.append(qid)

            if len(bin_qids) < 3:
                concept_result["bins"][bin_name] = {
                    "range": [lo, hi],
                    "n_queries": len(bin_qids),
                    "note": "too few queries for analysis",
                }
                continue

            base_scores = np.array([base_pq[qid]["ndcg_cut_10"] for qid in bin_qids])
            sub_scores = np.array([sub_pq[qid]["ndcg_cut_10"] for qid in bin_qids])
            deltas = sub_scores - base_scores

            # Statistical test within bin
            if np.any(deltas != 0):
                t_stat, t_pval = stats.ttest_rel(base_scores, sub_scores)
            else:
                t_stat, t_pval = 0.0, 1.0

            concept_result["bins"][bin_name] = {
                "range": [lo, hi],
                "n_queries": len(bin_qids),
                "base_ndcg_mean": float(base_scores.mean()),
                "sub_ndcg_mean": float(sub_scores.mean()),
                "delta_mean": float(deltas.mean()),
                "delta_std": float(deltas.std()),
                "t_pvalue": float(t_pval),
                "pct_improved": float((deltas > 0).mean()),
                "pct_degraded": float((deltas < 0).mean()),
                "pct_unchanged": float((deltas == 0).mean()),
            }

            sig_marker = "*" if t_pval < 0.05 else ""
            print(f"    [{lo:.1f}-{hi:.1f}] n={len(bin_qids):>4}  "
                  f"base={base_scores.mean():.4f}  sub={sub_scores.mean():.4f}  "
                  f"Δ={deltas.mean():+.4f}  p={t_pval:.3f}{sig_marker}")

        ds_results["concepts"][concept] = concept_result

    # ── Control: orthogonal random concepts ──
    print(f"\n  Control: random orthogonal concepts...")
    np.random.seed(42)
    dim = corpus_embs.shape[1]
    query_centroid = query_embs.mean(axis=0)
    query_centroid = query_centroid / np.linalg.norm(query_centroid)

    control_results = []
    for trial in range(5):
        # Generate random vector and make it orthogonal to query centroid
        rand_vec = np.random.randn(dim).astype(np.float32)
        rand_vec = rand_vec - np.dot(rand_vec, query_centroid) * query_centroid
        rand_vec = rand_vec / np.linalg.norm(rand_vec)

        # Verify orthogonality
        proj_to_centroid = abs(np.dot(rand_vec, query_centroid))

        # Measure projections
        projs = np.abs(query_embs @ rand_vec)

        # Subtract and evaluate
        sub_embs = np.array([subtract_orthogonal(q, rand_vec) for q in query_embs])
        sub_sims = sub_embs @ corpus_embs.T
        sub_results = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sub_sims[i])[::-1][:100]
            sub_results[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
        sub_pq = per_query_eval.evaluate(sub_results)

        base_arr = np.array([base_pq[qid]["ndcg_cut_10"] for qid in valid_qids])
        sub_arr = np.array([sub_pq[qid]["ndcg_cut_10"] for qid in valid_qids])
        delta = sub_arr - base_arr

        control_results.append({
            "trial": trial,
            "proj_to_centroid": float(proj_to_centroid),
            "mean_proj_to_queries": float(projs.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "pct_changed": float((delta != 0).mean()),
        })
        print(f"    Trial {trial}: mean|proj|={projs.mean():.4f}  "
              f"Δ={delta.mean():+.5f}  changed={((delta != 0).mean())*100:.1f}%")

    ds_results["control_orthogonal"] = control_results
    all_results[ds_name] = ds_results

# Save
with open(RESULTS_DIR / "threshold_analysis.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Summary
print(f"\n\n{'='*80}")
print(f"  SUMMARY — Item 11: Projection Threshold Analysis")
print(f"{'='*80}")
print(f"\nKey finding: Does |proj| < 0.2 predict safe subtraction?")
for ds, data in all_results.items():
    print(f"\n  {ds}:")
    for concept, cdata in data["concepts"].items():
        low_bins = []
        high_bins = []
        for bname, bdata in cdata["bins"].items():
            if "delta_mean" not in bdata:
                continue
            if bname in ("very_low", "low"):
                low_bins.append(bdata["delta_mean"])
            else:
                high_bins.append(bdata["delta_mean"])
        low_avg = np.mean(low_bins) if low_bins else float('nan')
        high_avg = np.mean(high_bins) if high_bins else float('nan')
        print(f"    '{concept[:30]}': low-proj Δ={low_avg:+.4f}  high-proj Δ={high_avg:+.4f}")

print(f"\nControl (orthogonal random): should show near-zero delta")
for ds, data in all_results.items():
    deltas = [c["delta_mean"] for c in data["control_orthogonal"]]
    print(f"  {ds}: mean Δ={np.mean(deltas):+.5f}")

print(f"\nResults saved to {RESULTS_DIR}/")
