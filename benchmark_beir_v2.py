"""
A²RAG Benchmark v2 on BEIR datasets.

Two evaluation modes:
1. Standard IR (nDCG@10, MAP, Recall) — for subtraction + baselines
2. Domain Shift Analysis — for rotation (new capability, not comparable via nDCG)

Baselines: identity, PRF (Rocchio), HyDE (via Ollama)

Usage:
    python benchmark_beir_v2.py --datasets scifact --quick
    python benchmark_beir_v2.py --datasets scifact fiqa nfcorpus --full
    python benchmark_beir_v2.py --no-hyde  # Skip HyDE (slow)

Author: Renato Aparecido Gomes
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests as _requests

import numpy as np
from sentence_transformers import SentenceTransformer

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from a2rag import rotate_toward, subtract_orthogonal


# ─── Config ───────────────────────────────────────────────────────────

DATASETS = ["scifact", "fiqa", "nfcorpus", "trec-covid", "arguana"]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10
RESULTS_DIR = Path("results/beir_v2")
DATA_DIR = Path("data/beir")
OLLAMA_MODEL = "gemma3:12b"

SUBTRACTION_CONCEPTS = {
    "scifact": ["methodology and statistical analysis", "animal model studies",
                "genetic analysis", "clinical trials"],
    "fiqa": ["cryptocurrency", "real estate investment",
             "retirement planning", "tax optimization"],
    "nfcorpus": ["dietary supplements", "weight loss and obesity",
                 "cancer treatment", "pediatric nutrition"],
    "trec-covid": ["treatment and therapeutics", "non-pharmaceutical interventions",
                   "transmission dynamics", "diagnostic testing"],
    "arguana": ["economic arguments", "moral and ethical reasoning",
                "legal precedent", "environmental impact"],
}


# ─── HyDE via Ollama ─────────────────────────────────────────────────

def hyde_generate(query: str, ollama_model: str = OLLAMA_MODEL) -> str:
    """Generate a hypothetical document answering the query using Ollama REST API."""
    prompt = (
        f"Please write a short scientific passage (3-4 sentences) that would "
        f"be a relevant answer to the following query. Write only the passage, "
        f"no preamble.\n\nQuery: {query}"
    )
    try:
        resp = _requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        # Remove thinking tags if present (qwen3 quirk)
        if "<think>" in text:
            parts = text.split("</think>")
            text = parts[-1].strip() if len(parts) > 1 else text
        return text if text else query
    except Exception as e:
        print(f"      HyDE error: {e}")
        return query  # Fallback to original query


def retrieve_with_hyde(queries: dict, model: SentenceTransformer,
                       corpus_embs: np.ndarray, doc_ids: list,
                       ollama_model: str = OLLAMA_MODEL,
                       top_k: int = 100) -> dict:
    """HyDE baseline: generate hypothetical doc, encode it, retrieve."""
    query_ids = list(queries.keys())
    print(f"    Generating {len(query_ids)} hypothetical documents via Ollama...")

    hyde_texts = []
    for i, qid in enumerate(query_ids):
        hyde_text = hyde_generate(queries[qid], ollama_model)
        hyde_texts.append(hyde_text)
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(query_ids)} done")

    # Encode hypothetical documents
    hyde_embs = model.encode(hyde_texts, normalize_embeddings=True, show_progress_bar=False)
    hyde_embs = np.array(hyde_embs)

    # Retrieve using hypothetical doc embeddings
    similarities = hyde_embs @ corpus_embs.T
    results = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        results[qid] = {
            doc_ids[idx]: float(similarities[i, idx]) for idx in top_indices
        }
    return results


# ─── Core retrieval ──────────────────────────────────────────────────

def retrieve(query_embs, corpus_embs, doc_ids, query_ids, top_k=100):
    similarities = query_embs @ corpus_embs.T
    results = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        results[qid] = {
            doc_ids[idx]: float(similarities[i, idx]) for idx in top_indices
        }
    return results


def retrieve_with_subtraction(query_embs, corpus_embs, doc_ids, query_ids,
                               exclude_emb, top_k=100):
    subtracted = np.array([subtract_orthogonal(q, exclude_emb) for q in query_embs])
    return retrieve(subtracted, corpus_embs, doc_ids, query_ids, top_k)


def retrieve_with_prf(query_embs, corpus_embs, doc_ids, query_ids,
                      top_k_prf=3, beta=0.4, top_k=100):
    similarities = query_embs @ corpus_embs.T
    results = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(similarities[i])[::-1][:top_k_prf]
        centroid = corpus_embs[top_indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        expanded = (1 - beta) * query_embs[i] + beta * centroid
        expanded = expanded / (np.linalg.norm(expanded) + 1e-10)
        new_sims = corpus_embs @ expanded
        top_final = np.argsort(new_sims)[::-1][:top_k]
        results[qid] = {doc_ids[idx]: float(new_sims[idx]) for idx in top_final}
    return results


def compute_proj_stats(query_embs, concept_emb):
    projections = np.abs(query_embs @ concept_emb)
    return {
        "mean": float(projections.mean()),
        "std": float(projections.std()),
        "min": float(projections.min()),
        "max": float(projections.max()),
        "pct_above_0.1": float((projections > 0.1).mean()),
    }


# ─── Rotation Analysis (separate from nDCG) ─────────────────────────

def analyze_rotation_effect(query_embs, corpus_embs, doc_ids, query_ids,
                            target_emb, alphas, baseline_results):
    """Analyze how rotation changes the result set (not nDCG).

    Measures:
    - Result set overlap with baseline (Jaccard@10)
    - Mean similarity shift
    - New documents surfaced (not in baseline top-10)
    """
    analysis = {}
    for alpha in alphas:
        rotated = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
        rot_results = retrieve(rotated, corpus_embs, doc_ids, query_ids, TOP_K)

        overlaps = []
        new_docs_count = []
        sim_shifts = []

        for qid in query_ids:
            base_set = set(list(baseline_results[qid].keys())[:TOP_K])
            rot_set = set(list(rot_results[qid].keys())[:TOP_K])

            jaccard = len(base_set & rot_set) / len(base_set | rot_set) if base_set | rot_set else 1.0
            overlaps.append(jaccard)
            new_docs_count.append(len(rot_set - base_set))

            # Mean similarity of rotated results vs baseline results
            rot_sims = list(rot_results[qid].values())[:TOP_K]
            base_sims = list(baseline_results[qid].values())[:TOP_K]
            sim_shifts.append(np.mean(rot_sims) - np.mean(base_sims))

        analysis[f"alpha={alpha}"] = {
            "mean_jaccard@10": float(np.mean(overlaps)),
            "mean_new_docs": float(np.mean(new_docs_count)),
            "mean_sim_shift": float(np.mean(sim_shifts)),
            "pct_queries_with_change": float(np.mean([o < 1.0 for o in overlaps])),
        }
    return analysis


# ─── Main ────────────────────────────────────────────────────────────

def run_dataset(dataset_name, model, use_hyde=True, quick=False):
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    corpus, queries, qrels = load_dataset(dataset_name)
    print(f"  Corpus: {len(corpus)} | Queries: {len(queries)} | "
          f"Judgments: {sum(len(v) for v in qrels.values())}")

    if quick:
        qids = list(queries.keys())[:50]
        queries = {q: queries[q] for q in qids}
        qrels = {q: qrels[q] for q in qids if q in qrels}
        print(f"  [QUICK] Limited to {len(queries)} queries")

    corpus_embs, doc_ids = encode_corpus(model, corpus)
    query_embs, query_ids = encode_queries(model, queries)
    evaluator = EvaluateRetrieval()
    results = {"dataset": dataset_name, "num_queries": len(queries),
               "num_docs": len(corpus), "model": MODEL_NAME}

    # ── Baseline ──
    print("\n  [1/4] Baseline...")
    baseline = retrieve(query_embs, corpus_embs, doc_ids, query_ids)
    ndcg, map_s, recall, prec = evaluator.evaluate(qrels, baseline, [1, 5, 10])
    results["baseline"] = {"ndcg": ndcg, "map": map_s, "recall": recall, "precision": prec}
    b_ndcg = ndcg.get("NDCG@10", 0)
    print(f"    nDCG@10={b_ndcg:.4f}  MAP@10={map_s.get('MAP@10',0):.4f}  "
          f"Recall@10={recall.get('Recall@10',0):.4f}")

    # ── PRF ──
    print("\n  [2/4] PRF baseline...")
    prf = retrieve_with_prf(query_embs, corpus_embs, doc_ids, query_ids)
    ndcg, map_s, recall, prec = evaluator.evaluate(qrels, prf, [1, 5, 10])
    results["prf"] = {"ndcg": ndcg, "map": map_s, "recall": recall, "precision": prec}
    print(f"    nDCG@10={ndcg.get('NDCG@10',0):.4f}  "
          f"Δ={ndcg.get('NDCG@10',0)-b_ndcg:+.4f}")

    # ── HyDE ──
    if use_hyde:
        print("\n  [2.5/4] HyDE baseline (Ollama)...")
        hyde = retrieve_with_hyde(queries, model, corpus_embs, doc_ids)
        ndcg, map_s, recall, prec = evaluator.evaluate(qrels, hyde, [1, 5, 10])
        results["hyde"] = {"ndcg": ndcg, "map": map_s, "recall": recall, "precision": prec}
        print(f"    nDCG@10={ndcg.get('NDCG@10',0):.4f}  "
              f"Δ={ndcg.get('NDCG@10',0)-b_ndcg:+.4f}")

    # ── Subtraction ──
    print("\n  [3/4] Subtraction experiments...")
    results["subtraction"] = {}
    concepts = SUBTRACTION_CONCEPTS.get(dataset_name, [])
    for concept in concepts:
        concept_emb = model.encode(concept, normalize_embeddings=True)
        proj_stats = compute_proj_stats(query_embs, concept_emb)
        sub = retrieve_with_subtraction(query_embs, corpus_embs, doc_ids,
                                         query_ids, concept_emb)
        ndcg, map_s, recall, prec = evaluator.evaluate(qrels, sub, [1, 5, 10])
        results["subtraction"][concept] = {
            "ndcg": ndcg, "map": map_s, "recall": recall,
            "precision": prec, "projection_stats": proj_stats,
        }
        delta = ndcg.get("NDCG@10", 0) - b_ndcg
        print(f"    ⊖ '{concept}': nDCG@10={ndcg.get('NDCG@10',0):.4f}  "
              f"Δ={delta:+.4f}  mean|proj|={proj_stats['mean']:.3f}")

    # ── Rotation Analysis ──
    print("\n  [4/4] Rotation analysis (domain shift)...")
    target_text = {
        "scifact": "clinical medicine and patient outcomes",
        "fiqa": "macroeconomics and monetary policy",
        "nfcorpus": "clinical medicine and disease treatment",
        "trec-covid": "general infectious disease epidemiology",
        "arguana": "legal reasoning and jurisprudence",
    }.get(dataset_name, "general knowledge")

    target_emb = model.encode(target_text, normalize_embeddings=True)
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    rot_analysis = analyze_rotation_effect(
        query_embs, corpus_embs, doc_ids, query_ids,
        target_emb, alphas, baseline
    )
    results["rotation_analysis"] = {
        "target": target_text, "data": rot_analysis,
    }

    # Also compute nDCG for rotation to show degradation curve
    results["rotation_ndcg"] = {}
    for alpha in alphas:
        rotated = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
        rot_res = retrieve(rotated, corpus_embs, doc_ids, query_ids)
        ndcg, map_s, recall, prec = evaluator.evaluate(qrels, rot_res, [1, 5, 10])
        results["rotation_ndcg"][f"alpha={alpha}"] = {
            "ndcg@10": ndcg.get("NDCG@10", 0),
            "recall@10": recall.get("Recall@10", 0),
        }

    for alpha_key, data in rot_analysis.items():
        ndcg_val = results["rotation_ndcg"].get(alpha_key, {}).get("ndcg@10", 0)
        print(f"    α={alpha_key.split('=')[1]}: Jaccard@10={data['mean_jaccard@10']:.3f}  "
              f"new_docs={data['mean_new_docs']:.1f}  "
              f"nDCG@10={ndcg_val:.4f}  "
              f"changed={data['pct_queries_with_change']:.0%}")

    return results


def load_dataset(dataset_name):
    data_path = DATA_DIR / dataset_name
    if not data_path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        beir_util.download_and_unzip(url, str(DATA_DIR))
    return GenericDataLoader(str(data_path)).load(split="test")


def encode_corpus(model, corpus, batch_size=256):
    doc_ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
             for d in doc_ids]
    print(f"  Encoding {len(texts)} documents...")
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True)
    return np.array(embs), doc_ids


def encode_queries(model, queries):
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]
    print(f"  Encoding {len(texts)} queries...")
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embs), qids


def main():
    parser = argparse.ArgumentParser(description="A²RAG BEIR Benchmark v2")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-hyde", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"A²RAG BEIR Benchmark v2")
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {args.datasets}")
    print(f"HyDE: {'OFF' if args.no_hyde else f'ON ({OLLAMA_MODEL})'}")

    model = SentenceTransformer(MODEL_NAME)
    all_results = {}

    for ds in args.datasets:
        try:
            r = run_dataset(ds, model, use_hyde=not args.no_hyde, quick=args.quick)
            all_results[ds] = r
            with open(RESULTS_DIR / f"{ds}.json", "w") as f:
                json.dump(r, f, indent=2)
        except Exception as e:
            print(f"\n  ERROR on {ds}: {e}")
            import traceback; traceback.print_exc()

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} {'Baseline':<10} {'PRF':<10} {'HyDE':<10} "
          f"{'Best Sub.':<10} {'Rot α=0.1':<12} {'Rot α=0.3':<12}")
    print("-" * 76)
    for ds, r in all_results.items():
        bl = r["baseline"]["ndcg"].get("NDCG@10", 0)
        prf = r["prf"]["ndcg"].get("NDCG@10", 0)
        hyde = r.get("hyde", {}).get("ndcg", {}).get("NDCG@10", 0)
        subs = [v["ndcg"].get("NDCG@10", 0) for v in r.get("subtraction", {}).values()]
        best_sub = max(subs) if subs else 0
        r01 = r.get("rotation_ndcg", {}).get("alpha=0.1", {}).get("ndcg@10", 0)
        r03 = r.get("rotation_ndcg", {}).get("alpha=0.3", {}).get("ndcg@10", 0)
        print(f"{ds:<12} {bl:<10.4f} {prf:<10.4f} {hyde:<10.4f} "
              f"{best_sub:<10.4f} {r01:<12.4f} {r03:<12.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
