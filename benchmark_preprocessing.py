"""
Item 10: Semantic preprocessing experiment.

Hypothesis: Rewriting documents in plain conceptual language makes embeddings
more isotropic, reducing unwanted projection and making subtraction safer.

Steps:
1. Take 100 SciFact documents
2. Rewrite with gemma3:12b (remove jargon, keep concepts)
3. Re-encode with BGE-small
4. Compare projection magnitudes and nDCG

Author: Renato Aparecido Gomes
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from a2rag import rotate_toward, subtract_orthogonal

MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
DATA_DIR = Path("data/beir/scifact")
RESULTS_DIR = Path("results/item10_preprocessing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DOCS = 100  # number of docs to rewrite

REWRITE_PROMPT = """Rewrite this scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use everyday language. Be concise — output only the rewritten text, nothing else.

Original:
{text}

Rewritten:"""

CONCEPTS = {
    "rotation_target": "clinical medicine and patient outcomes",
    "subtraction": [
        "methodology and statistical analysis",
        "animal model studies",
        "genetic analysis",
    ],
}


def rewrite_with_ollama(text: str, max_retries: int = 2) -> str:
    """Rewrite a document using Ollama gemma3:12b."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": REWRITE_PROMPT.format(text=text),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512},
            }, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt == max_retries:
                print(f"    FAILED after {max_retries+1} attempts: {e}")
                return text  # fallback to original
            time.sleep(2)


def evaluate_retrieval(query_embs, corpus_embs, doc_ids, query_ids, qrels):
    """Compute nDCG@10 for given embeddings."""
    evaluator = EvaluateRetrieval()
    sims = query_embs @ corpus_embs.T
    results = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(sims[i])[::-1][:100]
        results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
    ndcg, map_s, recall, prec = evaluator.evaluate(qrels, results, [1, 5, 10])
    return {
        "ndcg@10": ndcg.get("NDCG@10", 0),
        "ndcg@5": ndcg.get("NDCG@5", 0),
        "map@10": map_s.get("MAP@10", 0),
        "recall@10": recall.get("Recall@10", 0),
    }


def main():
    np.random.seed(42)

    # Load data
    print("Loading SciFact...")
    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR)).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Select N_DOCS docs that have relevance judgments
    relevant_docs = set()
    for qid, rels in qrels.items():
        for did, score in rels.items():
            if score > 0:
                relevant_docs.add(did)

    # Pick docs: first from relevant, then fill with random
    selected_idx = []
    for i, did in enumerate(doc_ids):
        if did in relevant_docs and len(selected_idx) < N_DOCS:
            selected_idx.append(i)
    # Fill remaining
    remaining = [i for i in range(len(doc_ids)) if i not in selected_idx]
    np.random.shuffle(remaining)
    selected_idx.extend(remaining[:max(0, N_DOCS - len(selected_idx))])
    selected_idx = sorted(selected_idx[:N_DOCS])

    print(f"Selected {len(selected_idx)} docs ({sum(1 for i in selected_idx if doc_ids[i] in relevant_docs)} with relevance judgments)")

    # Rewrite selected docs with gemma3
    print(f"\nRewriting {len(selected_idx)} docs with {OLLAMA_MODEL}...")
    rewritten_texts = {}
    cache_file = RESULTS_DIR / "rewritten_docs.json"

    if cache_file.exists():
        print("  Loading cached rewrites...")
        with open(cache_file) as f:
            rewritten_texts = json.load(f)
        print(f"  Loaded {len(rewritten_texts)} cached rewrites")

    t0 = time.time()
    for count, idx in enumerate(selected_idx):
        did = doc_ids[idx]
        if did in rewritten_texts:
            continue
        original = doc_texts[idx]
        rewritten = rewrite_with_ollama(original)
        rewritten_texts[did] = rewritten

        if (count + 1) % 10 == 0 or count == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{count+1}/{len(selected_idx)}] {rate:.1f} docs/s  |  "
                  f"orig: {len(original)} chars → rewritten: {len(rewritten)} chars")

        # Save periodically
        if (count + 1) % 20 == 0:
            with open(cache_file, "w") as f:
                json.dump(rewritten_texts, f, indent=2)

    rewrite_time = time.time() - t0
    with open(cache_file, "w") as f:
        json.dump(rewritten_texts, f, indent=2)
    print(f"  Done in {rewrite_time:.0f}s")

    # Build two corpora: original and rewritten (only for selected docs)
    orig_subset_texts = [doc_texts[i] for i in selected_idx]
    rewr_subset_texts = [rewritten_texts.get(doc_ids[i], doc_texts[i]) for i in selected_idx]
    subset_doc_ids = [doc_ids[i] for i in selected_idx]

    # Encode
    print(f"\nEncoding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    orig_embs = np.array(model.encode(orig_subset_texts, batch_size=256,
                                       normalize_embeddings=True, show_progress_bar=False))
    rewr_embs = np.array(model.encode(rewr_subset_texts, batch_size=256,
                                       normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                        show_progress_bar=False))

    # Also encode full corpus for retrieval evaluation
    all_corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                             normalize_embeddings=True, show_progress_bar=True))
    # Replace selected docs with rewritten versions
    rewr_full_embs = all_corpus_embs.copy()
    for i, idx in enumerate(selected_idx):
        rewr_full_embs[idx] = rewr_embs[i]

    results = {"config": {
        "model": MODEL_NAME, "ollama_model": OLLAMA_MODEL,
        "n_docs_rewritten": len(selected_idx), "total_docs": len(doc_ids),
    }}

    # 1. Compare embeddings: cosine similarity between original and rewritten
    cos_sims = np.sum(orig_embs * rewr_embs, axis=1)
    results["embedding_similarity"] = {
        "mean_cosine": float(cos_sims.mean()),
        "std_cosine": float(cos_sims.std()),
        "min_cosine": float(cos_sims.min()),
        "max_cosine": float(cos_sims.max()),
    }
    print(f"\nEmbedding similarity (orig vs rewritten): {cos_sims.mean():.4f} +/- {cos_sims.std():.4f}")

    # 2. Projection analysis per concept
    print(f"\nProjection analysis:")
    results["projections"] = {}
    for concept_name, concept_text in [("subtraction_0", CONCEPTS["subtraction"][0]),
                                         ("subtraction_1", CONCEPTS["subtraction"][1]),
                                         ("subtraction_2", CONCEPTS["subtraction"][2]),
                                         ("rotation_target", CONCEPTS["rotation_target"])]:
        concept_emb = model.encode(concept_text, normalize_embeddings=True)

        orig_proj = np.abs(orig_embs @ concept_emb)
        rewr_proj = np.abs(rewr_embs @ concept_emb)

        results["projections"][concept_name] = {
            "concept": concept_text,
            "original": {"mean": float(orig_proj.mean()), "std": float(orig_proj.std())},
            "rewritten": {"mean": float(rewr_proj.mean()), "std": float(rewr_proj.std())},
            "delta_mean": float(rewr_proj.mean() - orig_proj.mean()),
            "delta_pct": float((rewr_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100),
        }
        print(f"  '{concept_text[:40]}': orig={orig_proj.mean():.4f} → rewr={rewr_proj.mean():.4f} "
              f"({(rewr_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100:+.1f}%)")

    # 3. Retrieval evaluation: original corpus vs hybrid corpus (selected docs rewritten)
    print(f"\nRetrieval evaluation:")

    # Baseline on original
    baseline = evaluate_retrieval(query_embs, all_corpus_embs, doc_ids, query_ids, qrels)
    results["retrieval_baseline"] = baseline
    print(f"  Baseline (original corpus):   nDCG@10 = {baseline['ndcg@10']:.4f}")

    # Hybrid: rewritten selected docs
    hybrid = evaluate_retrieval(query_embs, rewr_full_embs, doc_ids, query_ids, qrels)
    results["retrieval_hybrid"] = hybrid
    print(f"  Hybrid (100 docs rewritten):  nDCG@10 = {hybrid['ndcg@10']:.4f} "
          f"({hybrid['ndcg@10'] - baseline['ndcg@10']:+.4f})")

    # 4. Rotation on hybrid vs original
    print(f"\nRotation (alpha=0.1) comparison:")
    target_emb = model.encode(CONCEPTS["rotation_target"], normalize_embeddings=True)

    rotated_q = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])

    rot_orig = evaluate_retrieval(rotated_q, all_corpus_embs, doc_ids, query_ids, qrels)
    rot_hybrid = evaluate_retrieval(rotated_q, rewr_full_embs, doc_ids, query_ids, qrels)

    results["rotation_0.1_original"] = rot_orig
    results["rotation_0.1_hybrid"] = rot_hybrid
    print(f"  Rot on original: nDCG@10 = {rot_orig['ndcg@10']:.4f} ({rot_orig['ndcg@10'] - baseline['ndcg@10']:+.4f})")
    print(f"  Rot on hybrid:   nDCG@10 = {rot_hybrid['ndcg@10']:.4f} ({rot_hybrid['ndcg@10'] - baseline['ndcg@10']:+.4f})")

    # 5. Subtraction on hybrid vs original
    print(f"\nSubtraction comparison:")
    for i, concept_text in enumerate(CONCEPTS["subtraction"]):
        concept_emb = model.encode(concept_text, normalize_embeddings=True)
        sub_q = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])

        sub_orig = evaluate_retrieval(sub_q, all_corpus_embs, doc_ids, query_ids, qrels)
        sub_hybrid = evaluate_retrieval(sub_q, rewr_full_embs, doc_ids, query_ids, qrels)

        results[f"subtraction_{i}_original"] = sub_orig
        results[f"subtraction_{i}_hybrid"] = sub_hybrid
        print(f"  '{concept_text[:30]}' on original: {sub_orig['ndcg@10']:.4f} ({sub_orig['ndcg@10'] - baseline['ndcg@10']:+.4f})")
        print(f"  '{concept_text[:30]}' on hybrid:   {sub_hybrid['ndcg@10']:.4f} ({sub_hybrid['ndcg@10'] - baseline['ndcg@10']:+.4f})")

    # 6. Isotropy comparison
    print(f"\nIsotropy (mean pairwise cosine):")
    n_pairs = 3000
    idx_a = np.random.randint(0, len(selected_idx), n_pairs)
    idx_b = np.random.randint(0, len(selected_idx), n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    orig_iso = float(np.sum(orig_embs[idx_a] * orig_embs[idx_b], axis=1).mean())
    rewr_iso = float(np.sum(rewr_embs[idx_a] * rewr_embs[idx_b], axis=1).mean())
    results["isotropy"] = {"original": orig_iso, "rewritten": rewr_iso}
    print(f"  Original: {orig_iso:.4f}  |  Rewritten: {rewr_iso:.4f}")

    # Save
    with open(RESULTS_DIR / "preprocessing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Item 10: Semantic Preprocessing")
    print(f"{'='*70}")
    print(f"  Embedding similarity (orig↔rewr): {results['embedding_similarity']['mean_cosine']:.4f}")
    print(f"  Isotropy: orig={orig_iso:.4f} → rewr={rewr_iso:.4f}")
    print(f"  Baseline nDCG@10: {baseline['ndcg@10']:.4f}")
    print(f"  Hybrid nDCG@10:   {hybrid['ndcg@10']:.4f} ({hybrid['ndcg@10'] - baseline['ndcg@10']:+.4f})")
    print(f"  Rotation benefit: orig={rot_orig['ndcg@10'] - baseline['ndcg@10']:+.4f}, hybrid={rot_hybrid['ndcg@10'] - baseline['ndcg@10']:+.4f}")
    hyp = "SUPPORTED" if rewr_iso < orig_iso else "NOT SUPPORTED"
    print(f"\n  Hypothesis (preprocessing → more isotropic): {hyp}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
