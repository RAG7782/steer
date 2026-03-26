"""
A²RAG Benchmark on BEIR datasets.

Evaluates algebraic retrieval (rotation + subtraction) against standard
retrieval on BEIR benchmarks with standard IR metrics (nDCG@10, MAP@10, Recall@10).

Usage:
    python benchmark_beir.py                    # Run all datasets
    python benchmark_beir.py --datasets scifact # Run single dataset
    python benchmark_beir.py --quick            # Quick test (1 dataset, fewer queries)

Author: Renato Aparecido Gomes
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# BEIR imports
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from a2rag import rotate_toward, subtract_orthogonal


# ─── Configuration ────────────────────────────────────────────────────

DATASETS = ["scifact", "fiqa", "nfcorpus", "trec-covid", "arguana"]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
RESULTS_DIR = Path("results/beir")
DATA_DIR = Path("data/beir")


# ─── Cross-domain rotation pairs ─────────────────────────────────────
# For each BEIR dataset, define domain rotation targets.
# These are natural-language descriptions of domains that are
# semantically adjacent to the dataset's primary domain.

ROTATION_TARGETS = {
    "scifact": [
        ("biomedical research", "clinical medicine"),
        ("biomedical research", "pharmacology and drug development"),
        ("biomedical research", "public health and epidemiology"),
    ],
    "fiqa": [
        ("personal finance", "macroeconomics and monetary policy"),
        ("personal finance", "corporate finance and accounting"),
        ("personal finance", "cryptocurrency and blockchain"),
    ],
    "nfcorpus": [
        ("nutrition and food science", "clinical medicine"),
        ("nutrition and food science", "sports science and exercise physiology"),
        ("nutrition and food science", "agriculture and food production"),
    ],
    "trec-covid": [
        ("COVID-19 research", "general infectious disease epidemiology"),
        ("COVID-19 research", "vaccine development and immunology"),
        ("COVID-19 research", "public health policy"),
    ],
    "arguana": [
        ("argumentation and debate", "legal reasoning and jurisprudence"),
        ("argumentation and debate", "philosophy and ethics"),
        ("argumentation and debate", "political science and policy analysis"),
    ],
}

# Subtraction concepts per dataset (concepts to exclude from queries)
SUBTRACTION_CONCEPTS = {
    "scifact": ["methodology and statistical analysis", "animal studies"],
    "fiqa": ["cryptocurrency", "real estate"],
    "nfcorpus": ["dietary supplements", "weight loss"],
    "trec-covid": ["treatment and therapeutics", "non-pharmaceutical interventions"],
    "arguana": ["ad hominem attacks", "economic arguments"],
}


# ─── Core Functions ──────────────────────────────────────────────────

def load_dataset(dataset_name: str) -> tuple:
    """Download and load a BEIR dataset."""
    data_path = DATA_DIR / dataset_name
    if not data_path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        beir_util.download_and_unzip(url, str(DATA_DIR))

    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")
    return corpus, queries, qrels


def encode_corpus(model: SentenceTransformer, corpus: dict, batch_size: int = 64) -> tuple:
    """Encode all corpus documents, return embeddings and ID mapping."""
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
        for did in doc_ids
    ]

    print(f"  Encoding {len(doc_texts)} documents...")
    embeddings = model.encode(doc_texts, batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings), doc_ids


def encode_queries(model: SentenceTransformer, queries: dict) -> tuple:
    """Encode all queries, return embeddings and ID mapping."""
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  Encoding {len(query_texts)} queries...")
    embeddings = model.encode(query_texts, normalize_embeddings=True,
                              show_progress_bar=True)
    return np.array(embeddings), query_ids


def retrieve(query_embs: np.ndarray, corpus_embs: np.ndarray,
             doc_ids: list, query_ids: list, top_k: int = 100) -> dict:
    """Standard cosine similarity retrieval. Returns BEIR-format results."""
    # Compute all similarities at once
    similarities = query_embs @ corpus_embs.T  # (num_queries, num_docs)

    results = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        results[qid] = {
            doc_ids[idx]: float(similarities[i, idx])
            for idx in top_indices
        }
    return results


def retrieve_with_rotation(query_embs: np.ndarray, corpus_embs: np.ndarray,
                           doc_ids: list, query_ids: list,
                           target_emb: np.ndarray, alpha: float,
                           top_k: int = 100) -> dict:
    """Retrieval with embedding rotation applied to all queries."""
    rotated = np.array([
        rotate_toward(q, target_emb, alpha) for q in query_embs
    ])
    return retrieve(rotated, corpus_embs, doc_ids, query_ids, top_k)


def retrieve_with_subtraction(query_embs: np.ndarray, corpus_embs: np.ndarray,
                               doc_ids: list, query_ids: list,
                               exclude_emb: np.ndarray,
                               top_k: int = 100) -> dict:
    """Retrieval with orthogonal subtraction applied to all queries."""
    subtracted = np.array([
        subtract_orthogonal(q, exclude_emb) for q in query_embs
    ])
    return retrieve(subtracted, corpus_embs, doc_ids, query_ids, top_k)


def retrieve_with_prf(query_embs: np.ndarray, corpus_embs: np.ndarray,
                      doc_ids: list, query_ids: list,
                      top_k_prf: int = 3, beta: float = 0.4,
                      top_k: int = 100) -> dict:
    """Pseudo-Relevance Feedback baseline (Rocchio-style).

    First retrieves top_k_prf docs, then re-queries with centroid.
    """
    similarities = query_embs @ corpus_embs.T

    results = {}
    for i, qid in enumerate(query_ids):
        # Initial retrieval
        top_indices = np.argsort(similarities[i])[::-1][:top_k_prf]

        # Compute centroid of top docs
        centroid = corpus_embs[top_indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

        # Rocchio: blend original query with centroid
        expanded = (1 - beta) * query_embs[i] + beta * centroid
        expanded = expanded / (np.linalg.norm(expanded) + 1e-10)

        # Re-retrieve
        new_sims = corpus_embs @ expanded
        top_final = np.argsort(new_sims)[::-1][:top_k]
        results[qid] = {
            doc_ids[idx]: float(new_sims[idx]) for idx in top_final
        }

    return results


def compute_projection_norms(query_embs: np.ndarray, concept_emb: np.ndarray) -> dict:
    """Compute projection magnitudes to assess transformation strength."""
    projections = np.array([
        abs(np.dot(q, concept_emb)) for q in query_embs
    ])
    return {
        "mean_proj_norm": float(projections.mean()),
        "std_proj_norm": float(projections.std()),
        "min_proj_norm": float(projections.min()),
        "max_proj_norm": float(projections.max()),
        "pct_above_0.05": float((projections > 0.05).mean()),
    }


# ─── Main Benchmark ──────────────────────────────────────────────────

def run_benchmark(dataset_name: str, model: SentenceTransformer,
                  quick: bool = False) -> dict:
    """Run full benchmark on one BEIR dataset."""
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    corpus, queries, qrels = load_dataset(dataset_name)
    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)} | "
          f"Relevant judgments: {sum(len(v) for v in qrels.values())}")

    if quick:
        # Limit to first 50 queries for quick testing
        query_ids_limited = list(queries.keys())[:50]
        queries = {qid: queries[qid] for qid in query_ids_limited}
        qrels = {qid: qrels[qid] for qid in query_ids_limited if qid in qrels}
        print(f"  [QUICK MODE] Limited to {len(queries)} queries")

    # Encode
    corpus_embs, doc_ids = encode_corpus(model, corpus)
    query_embs, query_ids = encode_queries(model, queries)

    evaluator = EvaluateRetrieval()
    all_results = {}

    # ── 1. Baseline (identity) ────────────────────────────────────
    print("\n  [Baseline] Standard retrieval...")
    t0 = time.time()
    baseline_results = retrieve(query_embs, corpus_embs, doc_ids, query_ids)
    baseline_time = time.time() - t0

    ndcg, map_score, recall, precision = evaluator.evaluate(
        qrels, baseline_results, [1, 5, 10]
    )
    all_results["baseline"] = {
        "ndcg": ndcg, "map": map_score, "recall": recall, "precision": precision,
        "time_seconds": baseline_time,
    }
    print(f"    nDCG@10={ndcg.get('NDCG@10', 0):.4f}  "
          f"MAP@10={map_score.get('MAP@10', 0):.4f}  "
          f"Recall@10={recall.get('Recall@10', 0):.4f}")

    # ── 2. Rotation experiments ───────────────────────────────────
    targets = ROTATION_TARGETS.get(dataset_name, [])
    for source_desc, target_desc in targets:
        target_emb = model.encode(target_desc, normalize_embeddings=True)

        for alpha in ALPHAS:
            key = f"rotation|{target_desc}|alpha={alpha}"
            print(f"\n  [Rotation] → '{target_desc}' (α={alpha})...")

            t0 = time.time()
            rot_results = retrieve_with_rotation(
                query_embs, corpus_embs, doc_ids, query_ids, target_emb, alpha
            )
            rot_time = time.time() - t0

            ndcg, map_score, recall, precision = evaluator.evaluate(
                qrels, rot_results, [1, 5, 10]
            )
            all_results[key] = {
                "ndcg": ndcg, "map": map_score, "recall": recall,
                "precision": precision, "time_seconds": rot_time,
                "alpha": alpha, "target": target_desc,
            }
            print(f"    nDCG@10={ndcg.get('NDCG@10', 0):.4f}  "
                  f"Δ={ndcg.get('NDCG@10', 0) - all_results['baseline']['ndcg'].get('NDCG@10', 0):+.4f}")

    # ── 3. Subtraction experiments ────────────────────────────────
    concepts = SUBTRACTION_CONCEPTS.get(dataset_name, [])
    for concept in concepts:
        concept_emb = model.encode(concept, normalize_embeddings=True)
        key = f"subtraction|{concept}"
        print(f"\n  [Subtraction] ⊖ '{concept}'...")

        # Compute projection norms (item 9 from review)
        proj_stats = compute_projection_norms(query_embs, concept_emb)

        t0 = time.time()
        sub_results = retrieve_with_subtraction(
            query_embs, corpus_embs, doc_ids, query_ids, concept_emb
        )
        sub_time = time.time() - t0

        ndcg, map_score, recall, precision = evaluator.evaluate(
            qrels, sub_results, [1, 5, 10]
        )
        all_results[key] = {
            "ndcg": ndcg, "map": map_score, "recall": recall,
            "precision": precision, "time_seconds": sub_time,
            "concept": concept, "projection_stats": proj_stats,
        }
        print(f"    nDCG@10={ndcg.get('NDCG@10', 0):.4f}  "
              f"Δ={ndcg.get('NDCG@10', 0) - all_results['baseline']['ndcg'].get('NDCG@10', 0):+.4f}  "
              f"mean|proj|={proj_stats['mean_proj_norm']:.4f}")

    # ── 4. PRF baseline ───────────────────────────────────────────
    print(f"\n  [PRF] Pseudo-Relevance Feedback (top-3, β=0.4)...")
    t0 = time.time()
    prf_results = retrieve_with_prf(query_embs, corpus_embs, doc_ids, query_ids)
    prf_time = time.time() - t0

    ndcg, map_score, recall, precision = evaluator.evaluate(
        qrels, prf_results, [1, 5, 10]
    )
    all_results["prf_baseline"] = {
        "ndcg": ndcg, "map": map_score, "recall": recall,
        "precision": precision, "time_seconds": prf_time,
    }
    print(f"    nDCG@10={ndcg.get('NDCG@10', 0):.4f}  "
          f"Δ={ndcg.get('NDCG@10', 0) - all_results['baseline']['ndcg'].get('NDCG@10', 0):+.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="A²RAG BEIR Benchmark")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="BEIR datasets to evaluate")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 dataset, 50 queries")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Sentence transformer model")
    args = parser.parse_args()

    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"A²RAG BEIR Benchmark")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")

    model = SentenceTransformer(args.model)

    all_dataset_results = {}
    for dataset_name in args.datasets:
        try:
            results = run_benchmark(dataset_name, model, quick=args.quick)
            all_dataset_results[dataset_name] = results

            # Save per-dataset
            outfile = RESULTS_DIR / f"{dataset_name}_results.json"
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved: {outfile}")

        except Exception as e:
            print(f"\n  ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined
    outfile = RESULTS_DIR / "all_results.json"
    with open(outfile, "w") as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\nAll results saved to {outfile}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY: nDCG@10 across all datasets")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Baseline':<10} {'Best Rot.':<10} {'Best Sub.':<10} {'PRF':<10}")
    print(f"{'-'*55}")

    for ds, results in all_dataset_results.items():
        baseline = results.get("baseline", {}).get("ndcg", {}).get("NDCG@10", 0)

        # Best rotation
        rot_scores = [
            v["ndcg"].get("NDCG@10", 0)
            for k, v in results.items() if k.startswith("rotation|")
        ]
        best_rot = max(rot_scores) if rot_scores else 0

        # Best subtraction
        sub_scores = [
            v["ndcg"].get("NDCG@10", 0)
            for k, v in results.items() if k.startswith("subtraction|")
        ]
        best_sub = max(sub_scores) if sub_scores else 0

        prf = results.get("prf_baseline", {}).get("ndcg", {}).get("NDCG@10", 0)

        print(f"{ds:<15} {baseline:<10.4f} {best_rot:<10.4f} {best_sub:<10.4f} {prf:<10.4f}")


if __name__ == "__main__":
    main()
