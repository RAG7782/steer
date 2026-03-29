"""
Item 12: Multi-model benchmark for A²RAG.

Tests 6 embedding models on SciFact + ArguAna:
- nDCG@10 baseline vs rotation (α=0.1) vs subtraction
- Isotropy measurement (mean pairwise cosine similarity of random pairs)

Author: Renato Aparecido Gomes
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from a2rag import rotate_toward, subtract_orthogonal

DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/item12_multimodel")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
SUBTRACTION_CONCEPTS = {
    "scifact": "methodology and statistical analysis",
    "arguana": "economic arguments",
}


def measure_isotropy(embs: np.ndarray, n_pairs: int = 5000) -> dict:
    """Measure isotropy via mean pairwise cosine similarity of random pairs.

    Perfect isotropy → mean cosine ≈ 0 (embeddings uniformly distributed).
    Anisotropic → mean cosine >> 0 (embeddings clustered).
    """
    n = len(embs)
    idx_a = np.random.randint(0, n, size=n_pairs)
    idx_b = np.random.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    cos_sims = np.sum(embs[idx_a] * embs[idx_b], axis=1)
    return {
        "mean_cosine": float(cos_sims.mean()),
        "std_cosine": float(cos_sims.std()),
        "median_cosine": float(np.median(cos_sims)),
        "n_pairs": int(len(cos_sims)),
    }


def run_model_dataset(model_name: str, model: SentenceTransformer,
                       ds_name: str) -> dict:
    """Run baseline, rotation, and subtraction for one model on one dataset."""
    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR / ds_name)).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Encode
    t0 = time.time()
    corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                         normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                        show_progress_bar=False))
    encode_time = time.time() - t0

    evaluator = EvaluateRetrieval()
    result = {
        "model": model_name,
        "dataset": ds_name,
        "num_docs": len(doc_ids),
        "num_queries": len(query_ids),
        "embedding_dim": int(corpus_embs.shape[1]),
        "encode_time_s": round(encode_time, 1),
    }

    # Isotropy
    isotropy = measure_isotropy(corpus_embs)
    result["isotropy"] = isotropy

    # Baseline
    sims = query_embs @ corpus_embs.T
    base_res = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(sims[i])[::-1][:100]
        base_res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
    ndcg, map_s, recall, prec = evaluator.evaluate(qrels, base_res, [1, 5, 10])
    result["baseline"] = {
        "ndcg@10": ndcg.get("NDCG@10", 0),
        "ndcg@5": ndcg.get("NDCG@5", 0),
        "map@10": map_s.get("MAP@10", 0),
        "recall@10": recall.get("Recall@10", 0),
    }

    # Rotation α=0.1
    target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)
    rotated = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])
    rot_sims = rotated @ corpus_embs.T
    rot_res = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(rot_sims[i])[::-1][:100]
        rot_res[qid] = {doc_ids[idx]: float(rot_sims[i, idx]) for idx in top}
    ndcg, _, _, _ = evaluator.evaluate(qrels, rot_res, [10])
    rot_ndcg = ndcg.get("NDCG@10", 0)
    result["rotation_0.1"] = {
        "ndcg@10": rot_ndcg,
        "delta": round(rot_ndcg - result["baseline"]["ndcg@10"], 4),
        "delta_pct": round((rot_ndcg - result["baseline"]["ndcg@10"]) / result["baseline"]["ndcg@10"] * 100, 2),
    }

    # Rotation α=0.2
    rotated2 = np.array([rotate_toward(q, target_emb, 0.2) for q in query_embs])
    rot2_sims = rotated2 @ corpus_embs.T
    rot2_res = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(rot2_sims[i])[::-1][:100]
        rot2_res[qid] = {doc_ids[idx]: float(rot2_sims[i, idx]) for idx in top}
    ndcg2, _, _, _ = evaluator.evaluate(qrels, rot2_res, [10])
    rot2_ndcg = ndcg2.get("NDCG@10", 0)
    result["rotation_0.2"] = {
        "ndcg@10": rot2_ndcg,
        "delta": round(rot2_ndcg - result["baseline"]["ndcg@10"], 4),
        "delta_pct": round((rot2_ndcg - result["baseline"]["ndcg@10"]) / result["baseline"]["ndcg@10"] * 100, 2),
    }

    # Subtraction
    concept_emb = model.encode(SUBTRACTION_CONCEPTS[ds_name], normalize_embeddings=True)
    sub = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
    sub_sims = sub @ corpus_embs.T
    sub_res = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(sub_sims[i])[::-1][:100]
        sub_res[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
    ndcg, _, _, _ = evaluator.evaluate(qrels, sub_res, [10])
    sub_ndcg = ndcg.get("NDCG@10", 0)
    result["subtraction"] = {
        "ndcg@10": sub_ndcg,
        "delta": round(sub_ndcg - result["baseline"]["ndcg@10"], 4),
        "delta_pct": round((sub_ndcg - result["baseline"]["ndcg@10"]) / result["baseline"]["ndcg@10"] * 100, 2),
    }

    # Projection stats for subtraction concept
    proj_norms = np.abs(query_embs @ concept_emb)
    result["projection_stats"] = {
        "mean": float(proj_norms.mean()),
        "std": float(proj_norms.std()),
        "pct_above_0.2": float((proj_norms > 0.2).mean()),
    }

    return result


def load_existing_results() -> dict:
    """Load previously saved incremental results."""
    results = {}
    for f in RESULTS_DIR.glob("model_*.json"):
        with open(f) as fp:
            data = json.load(fp)
            model_name = data.pop("_model_name")
            results[model_name] = data
    return results


def save_model_result(model_name: str, model_results: dict):
    """Save one model's results incrementally to its own file."""
    safe_name = model_name.replace("/", "__")
    data = {"_model_name": model_name, **model_results}
    with open(RESULTS_DIR / f"model_{safe_name}.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    np.random.seed(42)
    all_results = load_existing_results()

    for model_name, params_m, family in MODELS:
        if model_name in all_results:
            print(f"\n  SKIP (already completed): {model_name}")
            continue

        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name} ({params_m}M, {family})")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name)
        model_results = {"params_M": params_m, "family": family}

        for ds in DATASETS:
            print(f"\n  Dataset: {ds}")
            r = run_model_dataset(model_name, model, ds)
            model_results[ds] = r

            bl = r["baseline"]["ndcg@10"]
            rot = r["rotation_0.1"]["ndcg@10"]
            sub = r["subtraction"]["ndcg@10"]
            iso = r["isotropy"]["mean_cosine"]
            print(f"    Baseline: {bl:.4f} | Rot α=0.1: {rot:.4f} ({r['rotation_0.1']['delta']:+.4f}) "
                  f"| Sub: {sub:.4f} ({r['subtraction']['delta']:+.4f}) | Isotropy: {iso:.4f}")

        # Save incrementally after each model completes
        save_model_result(model_name, model_results)
        all_results[model_name] = model_results
        print(f"  ✓ Saved results for {model_name}")

        # Free memory
        del model

    # Save combined results
    with open(RESULTS_DIR / "multimodel_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY — Item 12: Multi-Model Benchmark")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Params':>6} {'Family':<18} {'Dataset':<10} "
          f"{'Baseline':>8} {'Rot 0.1':>8} {'Rot 0.2':>8} {'Sub':>8} {'Isotropy':>8}")
    print("-" * 110)

    for model_name, data in all_results.items():
        for ds in DATASETS:
            r = data[ds]
            short_name = model_name.split("/")[-1]
            print(f"{short_name:<30} {data['params_M']:>4}M  {data['family']:<18} {ds:<10} "
                  f"{r['baseline']['ndcg@10']:>8.4f} {r['rotation_0.1']['ndcg@10']:>8.4f} "
                  f"{r['rotation_0.2']['ndcg@10']:>8.4f} {r['subtraction']['ndcg@10']:>8.4f} "
                  f"{r['isotropy']['mean_cosine']:>8.4f}")

    # Isotropy vs Delta analysis
    print(f"\n\n  ISOTROPY vs ROTATION DELTA (α=0.1)")
    print(f"{'Model':<30} {'Isotropy':>8} {'SciFact Δ':>10} {'ArguAna Δ':>10}")
    print("-" * 60)
    for model_name, data in all_results.items():
        short_name = model_name.split("/")[-1]
        iso_sf = data["scifact"]["isotropy"]["mean_cosine"]
        iso_ar = data["arguana"]["isotropy"]["mean_cosine"]
        avg_iso = (iso_sf + iso_ar) / 2
        d_sf = data["scifact"]["rotation_0.1"]["delta"]
        d_ar = data["arguana"]["rotation_0.1"]["delta"]
        print(f"{short_name:<30} {avg_iso:>8.4f} {d_sf:>+10.4f} {d_ar:>+10.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
