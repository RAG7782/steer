"""
A²RAG Deep Analysis — Items 1-5, 7 from review improvement plan.

Computes:
1. Statistical significance (paired t-test, Wilcoxon, confidence intervals)
2. Per-query analysis (which queries benefit from rotation)
3. Pareto frontier data (nDCG vs new docs)
4. Angular distance θ between queries and rotation targets
5. Projection norm vs nDCG correlation (threshold analysis)
7. Composition of operations (rotate after subtract)

Author: Renato Aparecido Gomes
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

from a2rag import rotate_toward, subtract_orthogonal
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/deep_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}

SUBTRACTION_CONCEPTS = {
    "scifact": ["methodology and statistical analysis", "animal model studies",
                "genetic analysis", "clinical trials"],
    "arguana": ["economic arguments", "moral and ethical reasoning",
                "legal precedent", "environmental impact"],
}

DATASETS = ["scifact", "arguana"]  # Smaller datasets for fast analysis

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

all_results = {}

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    # Load data
    data_path = DATA_DIR / ds_name
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")

    # Encode
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    print(f"  Encoding {len(doc_texts)} docs, {len(query_texts)} queries...")
    corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True))

    evaluator = EvaluateRetrieval()
    ds_results = {"dataset": ds_name, "num_queries": len(query_ids)}

    # ── Baseline per-query ────────────────────────────────────────
    print("\n  Computing baseline per-query scores...")
    base_sims = query_embs @ corpus_embs.T
    base_results = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argsort(base_sims[i])[::-1][:100]
        base_results[qid] = {doc_ids[idx]: float(base_sims[i, idx]) for idx in top_idx}

    # Get per-query nDCG
    from pytrec_eval import RelevanceEvaluator
    per_query_eval = RelevanceEvaluator(qrels, {"ndcg_cut.10", "map_cut.10", "recall.10"})
    base_per_query = per_query_eval.evaluate(base_results)

    # ── Item 4: Angular distance θ ────────────────────────────────
    print("\n  [Item 4] Angular distance to rotation target...")
    target_text = ROTATION_TARGETS[ds_name]
    target_emb = model.encode(target_text, normalize_embeddings=True)

    # Cosine similarity = cos(θ), so θ = arccos(sim)
    cos_sims = query_embs @ target_emb
    thetas_rad = np.arccos(np.clip(cos_sims, -1, 1))
    thetas_deg = np.degrees(thetas_rad)

    ds_results["angular_distance"] = {
        "target": target_text,
        "mean_theta_deg": float(thetas_deg.mean()),
        "std_theta_deg": float(thetas_deg.std()),
        "min_theta_deg": float(thetas_deg.min()),
        "max_theta_deg": float(thetas_deg.max()),
        "mean_cosine_sim": float(cos_sims.mean()),
    }
    print(f"    θ mean={thetas_deg.mean():.1f}° std={thetas_deg.std():.1f}° "
          f"range=[{thetas_deg.min():.1f}°, {thetas_deg.max():.1f}°]")
    print(f"    cos(sim) mean={cos_sims.mean():.3f}")

    # NLERP vs SLERP error estimate at mean θ
    mean_theta = thetas_rad.mean()
    alpha = 0.2
    # SLERP point
    slerp_val = np.sin((1-alpha)*mean_theta)/np.sin(mean_theta) * 1.0 + \
                np.sin(alpha*mean_theta)/np.sin(mean_theta) * 1.0  # scalar approx
    nlerp_val = (1-alpha) + alpha  # = 1.0 before normalization
    # The actual error is in direction, not magnitude. Approximate:
    nlerp_angle = alpha * mean_theta  # NLERP traverses chord
    slerp_angle = alpha * mean_theta  # SLERP traverses arc (same for small θ)
    # For moderate θ, the relative error is ~ (1 - cos(θ/2)) / cos(θ/2)
    approx_error_pct = (1 - np.cos(mean_theta * alpha)) * 100
    ds_results["nlerp_vs_slerp_error_pct"] = float(approx_error_pct)
    print(f"    NLERP vs SLERP approx error at α={alpha}: {approx_error_pct:.2f}%")

    # ── Item 1: Statistical significance ──────────────────────────
    print("\n  [Item 1] Statistical significance...")

    alphas_to_test = [0.1, 0.2, 0.3]
    ds_results["significance"] = {}

    for alpha in alphas_to_test:
        # Compute rotated per-query
        rotated_embs = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
        rot_sims = rotated_embs @ corpus_embs.T
        rot_results = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(rot_sims[i])[::-1][:100]
            rot_results[qid] = {doc_ids[idx]: float(rot_sims[i, idx]) for idx in top_idx}

        rot_per_query = per_query_eval.evaluate(rot_results)

        # Paired comparison
        base_scores = [base_per_query[qid]["ndcg_cut_10"] for qid in query_ids if qid in base_per_query]
        rot_scores = [rot_per_query[qid]["ndcg_cut_10"] for qid in query_ids if qid in rot_per_query]

        # Filter to queries that have qrels
        valid_qids = [qid for qid in query_ids if qid in base_per_query and qid in rot_per_query]
        base_arr = np.array([base_per_query[qid]["ndcg_cut_10"] for qid in valid_qids])
        rot_arr = np.array([rot_per_query[qid]["ndcg_cut_10"] for qid in valid_qids])
        diff = rot_arr - base_arr

        t_stat, t_pval = stats.ttest_rel(base_arr, rot_arr)
        w_stat, w_pval = stats.wilcoxon(diff[diff != 0]) if np.any(diff != 0) else (0, 1.0)

        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))

        improved = (diff > 0).sum()
        unchanged = (diff == 0).sum()
        degraded = (diff < 0).sum()

        key = f"rotation_alpha={alpha}"
        ds_results["significance"][key] = {
            "n_queries": len(valid_qids),
            "mean_delta": float(diff.mean()),
            "std_delta": float(diff.std()),
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "wilcoxon_pvalue": float(w_pval),
            "improved": int(improved),
            "unchanged": int(unchanged),
            "degraded": int(degraded),
        }
        print(f"    α={alpha}: Δ={diff.mean():+.4f} CI95=[{ci_95[0]:+.4f}, {ci_95[1]:+.4f}] "
              f"p={t_pval:.2e} (t-test) p={w_pval:.2e} (Wilcoxon) "
              f"↑{improved}/={unchanged}/↓{degraded}")

    # Subtraction significance
    for concept in SUBTRACTION_CONCEPTS[ds_name]:
        concept_emb = model.encode(concept, normalize_embeddings=True)
        sub_embs = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
        sub_sims = sub_embs @ corpus_embs.T
        sub_results = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(sub_sims[i])[::-1][:100]
            sub_results[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top_idx}

        sub_per_query = per_query_eval.evaluate(sub_results)
        valid_qids = [qid for qid in query_ids if qid in base_per_query and qid in sub_per_query]
        base_arr = np.array([base_per_query[qid]["ndcg_cut_10"] for qid in valid_qids])
        sub_arr = np.array([sub_per_query[qid]["ndcg_cut_10"] for qid in valid_qids])
        diff = sub_arr - base_arr

        t_stat, t_pval = stats.ttest_rel(base_arr, sub_arr)

        # Item 5: correlation with projection norm
        proj_norms = np.array([abs(np.dot(query_embs[query_ids.index(qid)], concept_emb))
                               for qid in valid_qids])

        # Bin by projection norm
        low_mask = proj_norms < 0.3
        mid_mask = (proj_norms >= 0.3) & (proj_norms < 0.5)
        high_mask = proj_norms >= 0.5

        key = f"subtraction_{concept[:20]}"
        ds_results["significance"][key] = {
            "n_queries": len(valid_qids),
            "mean_delta": float(diff.mean()),
            "t_pvalue": float(t_pval),
            "proj_norm_correlation": float(np.corrcoef(proj_norms, diff)[0,1]),
            "low_proj_delta": float(diff[low_mask].mean()) if low_mask.any() else None,
            "mid_proj_delta": float(diff[mid_mask].mean()) if mid_mask.any() else None,
            "high_proj_delta": float(diff[high_mask].mean()) if high_mask.any() else None,
            "low_proj_count": int(low_mask.sum()),
            "mid_proj_count": int(mid_mask.sum()),
            "high_proj_count": int(high_mask.sum()),
        }
        print(f"    ⊖ '{concept[:25]}': Δ={diff.mean():+.4f} p={t_pval:.2e} "
              f"corr(proj,Δ)={np.corrcoef(proj_norms, diff)[0,1]:+.3f}")

    # ── Item 2: Per-query rotation analysis ───────────────────────
    print("\n  [Item 2] Per-query rotation analysis (α=0.1)...")
    rotated_01 = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])
    rot01_sims = rotated_01 @ corpus_embs.T
    rot01_results = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argsort(rot01_sims[i])[::-1][:100]
        rot01_results[qid] = {doc_ids[idx]: float(rot01_sims[i, idx]) for idx in top_idx}

    rot01_pq = per_query_eval.evaluate(rot01_results)
    valid_qids = [qid for qid in query_ids if qid in base_per_query and qid in rot01_pq]

    diffs_01 = []
    query_lengths = []
    query_thetas = []
    for qid in valid_qids:
        d = rot01_pq[qid]["ndcg_cut_10"] - base_per_query[qid]["ndcg_cut_10"]
        diffs_01.append(d)
        query_lengths.append(len(queries[qid].split()))
        idx = query_ids.index(qid)
        query_thetas.append(float(thetas_deg[idx]))

    diffs_01 = np.array(diffs_01)
    query_lengths = np.array(query_lengths)
    query_thetas = np.array(query_thetas)

    # Correlations
    corr_length = np.corrcoef(query_lengths, diffs_01)[0,1]
    corr_theta = np.corrcoef(query_thetas, diffs_01)[0,1]

    improved_queries = (diffs_01 > 0).sum()
    degraded_queries = (diffs_01 < 0).sum()
    unchanged_queries = (diffs_01 == 0).sum()

    ds_results["per_query"] = {
        "alpha": 0.1,
        "improved": int(improved_queries),
        "degraded": int(degraded_queries),
        "unchanged": int(unchanged_queries),
        "mean_improvement_when_improved": float(diffs_01[diffs_01 > 0].mean()) if improved_queries > 0 else 0,
        "mean_degradation_when_degraded": float(diffs_01[diffs_01 < 0].mean()) if degraded_queries > 0 else 0,
        "correlation_query_length_vs_delta": float(corr_length),
        "correlation_theta_vs_delta": float(corr_theta),
    }
    print(f"    ↑{improved_queries} ={unchanged_queries} ↓{degraded_queries}")
    print(f"    corr(length, Δ)={corr_length:+.3f}  corr(θ, Δ)={corr_theta:+.3f}")
    if improved_queries > 0:
        print(f"    mean gain when improved: +{diffs_01[diffs_01 > 0].mean():.4f}")
    if degraded_queries > 0:
        print(f"    mean loss when degraded: {diffs_01[diffs_01 < 0].mean():.4f}")

    # ── Item 7: Composition of operations ─────────────────────────
    print("\n  [Item 7] Composition: subtract then rotate...")
    concept_emb = model.encode(SUBTRACTION_CONCEPTS[ds_name][0], normalize_embeddings=True)

    # Three conditions: subtract only, rotate only, subtract+rotate
    sub_only = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
    rot_only = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])
    composed = np.array([rotate_toward(subtract_orthogonal(q, concept_emb), target_emb, 0.1) for q in query_embs])

    conditions = {"subtract_only": sub_only, "rotate_only": rot_only, "composed": composed}
    ds_results["composition"] = {"concept": SUBTRACTION_CONCEPTS[ds_name][0], "target": target_text, "alpha": 0.1}

    for name, embs in conditions.items():
        sims = embs @ corpus_embs.T
        results = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(sims[i])[::-1][:100]
            results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top_idx}

        ndcg, map_s, recall, prec = evaluator.evaluate(qrels, results, [10])

        # Jaccard vs baseline
        jaccards = []
        for qid in query_ids:
            base_set = set(list(base_results[qid].keys())[:10])
            new_set = set(list(results[qid].keys())[:10])
            j = len(base_set & new_set) / len(base_set | new_set) if base_set | new_set else 1
            jaccards.append(j)

        ds_results["composition"][name] = {
            "ndcg@10": ndcg.get("NDCG@10", 0),
            "mean_jaccard@10": float(np.mean(jaccards)),
        }
        print(f"    {name}: nDCG@10={ndcg.get('NDCG@10',0):.4f} Jaccard@10={np.mean(jaccards):.3f}")

    all_results[ds_name] = ds_results

# ── Item 3: Pareto frontier data (already in benchmark results) ───
print("\n\n[Item 3] Pareto frontier data (from benchmark):")
print("α     | new_docs | nDCG@10 | nDCG_retained")
print("------|----------|---------|-------------")
pareto_data = [
    (0.05, 0.6, 0.503), (0.10, 1.0, 0.498), (0.15, 1.5, 0.488),
    (0.20, 2.0, 0.476), (0.30, 3.3, 0.442), (0.40, 5.0, 0.386), (0.50, 7.1, 0.297),
]
baseline_ndcg = 0.509  # average across 4 datasets
for alpha, new_docs, ndcg in pareto_data:
    retained = ndcg / baseline_ndcg * 100
    print(f"{alpha:.2f}  | {new_docs:.1f}      | {ndcg:.3f}  | {retained:.1f}%")

all_results["pareto"] = [{"alpha": a, "new_docs": n, "ndcg": d, "retained_pct": d/baseline_ndcg*100} for a,n,d in pareto_data]

# Save all
with open(RESULTS_DIR / "deep_analysis.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {RESULTS_DIR}/deep_analysis.json")
