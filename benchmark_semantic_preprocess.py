"""
Item 10: Semantic preprocessing experiment.

1. Take 100 SciFact documents
2. Rewrite each with gemma3:12b (conceptual language, no jargon)
3. Re-encode with BGE-small
4. Compare projections and nDCG between original and rewritten corpora

Hypothesis: rewritten docs will have smaller projections (more orthogonal concepts)
and subtraction will degrade less.

Author: Renato Aparecido Gomes
"""

import json
import time
import numpy as np
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from pytrec_eval import RelevanceEvaluator
from a2rag import subtract_orthogonal, rotate_toward

MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "gemma3:12b"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/item10_semantic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DOCS = 100  # Number of documents to rewrite

SUBTRACTION_CONCEPTS = ["methodology and statistical analysis", "animal model studies",
                        "genetic analysis", "clinical trials"]
ROTATION_TARGET = "clinical medicine and patient outcomes"

REWRITE_PROMPT = """Rewrite the following scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use only everyday language. Write ONLY the rewritten text, nothing else.

Original:
{text}

Rewritten:"""


def rewrite_with_ollama(text: str, max_retries: int = 2) -> str:
    """Rewrite text using Ollama gemma3:12b."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": REWRITE_PROMPT.format(text=text),
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300},
                },
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            if result:
                return result
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"    Rewrite failed: {e}")
    return text  # Fallback to original


def main():
    print(f"Item 10: Semantic Preprocessing Experiment")
    print(f"Model: {MODEL_NAME} | Rewriter: {OLLAMA_MODEL} | N docs: {N_DOCS}")

    # Warm up Ollama
    print("\nWarming up Ollama...")
    requests.post("http://localhost:11434/api/generate",
                   json={"model": OLLAMA_MODEL, "prompt": "Hello", "stream": False},
                   timeout=60)

    # Load SciFact
    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR / "scifact")).load(split="test")
    all_doc_ids = list(corpus.keys())

    # Select documents that appear in qrels (relevant docs)
    relevant_doc_ids = set()
    for qid_rels in qrels.values():
        for did, rel in qid_rels.items():
            if rel > 0:
                relevant_doc_ids.add(did)

    # Prioritize relevant docs, then fill with others
    selected_ids = list(relevant_doc_ids)[:N_DOCS]
    if len(selected_ids) < N_DOCS:
        remaining = [d for d in all_doc_ids if d not in relevant_doc_ids]
        selected_ids.extend(remaining[:N_DOCS - len(selected_ids)])
    selected_ids = selected_ids[:N_DOCS]

    print(f"\nSelected {len(selected_ids)} docs ({len([d for d in selected_ids if d in relevant_doc_ids])} relevant)")

    # Get original texts
    original_texts = {}
    for did in selected_ids:
        title = corpus[did].get("title", "")
        text = corpus[did].get("text", "")
        original_texts[did] = (title + " " + text).strip()

    # Rewrite documents
    print(f"\nRewriting {len(selected_ids)} documents with {OLLAMA_MODEL}...")
    rewritten_texts = {}
    rewrite_cache_path = RESULTS_DIR / "rewritten_docs.json"

    if rewrite_cache_path.exists():
        print("  Loading cached rewrites...")
        with open(rewrite_cache_path) as f:
            rewritten_texts = json.load(f)
        # Only rewrite missing ones
        to_rewrite = [did for did in selected_ids if did not in rewritten_texts]
        print(f"  {len(rewritten_texts)} cached, {len(to_rewrite)} to rewrite")
    else:
        to_rewrite = selected_ids

    for i, did in enumerate(to_rewrite):
        rewritten = rewrite_with_ollama(original_texts[did])
        rewritten_texts[did] = rewritten
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(to_rewrite)} done")
            # Save intermediate
            with open(rewrite_cache_path, "w") as f:
                json.dump(rewritten_texts, f, indent=2)

    # Final save
    with open(rewrite_cache_path, "w") as f:
        json.dump(rewritten_texts, f, indent=2)
    print(f"  All {len(rewritten_texts)} rewrites done/cached")

    # Show examples
    print(f"\n  Example rewrites:")
    for did in selected_ids[:3]:
        orig = original_texts[did][:150]
        rew = rewritten_texts[did][:150]
        print(f"    [{did}] Original: {orig}...")
        print(f"    [{did}] Rewritten: {rew}...")
        print()

    # Encode both corpora
    print("Encoding original and rewritten corpora...")
    model = SentenceTransformer(MODEL_NAME)

    orig_list = [original_texts[did] for did in selected_ids]
    rew_list = [rewritten_texts[did] for did in selected_ids]

    orig_embs = np.array(model.encode(orig_list, batch_size=256,
                                       normalize_embeddings=True, show_progress_bar=False))
    rew_embs = np.array(model.encode(rew_list, batch_size=256,
                                      normalize_embeddings=True, show_progress_bar=False))

    # Encode full corpus for retrieval eval (replace selected docs with rewritten versions)
    print("Encoding full corpus (with rewritten subset)...")
    full_orig_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                       for d in all_doc_ids]
    full_orig_embs = np.array(model.encode(full_orig_texts, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=True))

    # Create rewritten corpus (replace selected docs)
    full_rew_texts = list(full_orig_texts)
    for did in selected_ids:
        idx = all_doc_ids.index(did)
        full_rew_texts[idx] = rewritten_texts[did]
    full_rew_embs = np.array(model.encode(full_rew_texts, batch_size=256,
                                           normalize_embeddings=True, show_progress_bar=True))

    query_ids = list(queries.keys())
    query_texts_list = [queries[q] for q in query_ids]
    query_embs = np.array(model.encode(query_texts_list, normalize_embeddings=True))

    results = {"dataset": "scifact", "n_docs_rewritten": N_DOCS, "model": MODEL_NAME}

    # ── Compare projections ──
    print(f"\n{'='*60}")
    print(f"  PROJECTION COMPARISON")
    print(f"{'='*60}")

    results["projections"] = {}
    for concept in SUBTRACTION_CONCEPTS:
        concept_emb = model.encode(concept, normalize_embeddings=True)

        orig_proj = np.abs(orig_embs @ concept_emb)
        rew_proj = np.abs(rew_embs @ concept_emb)

        results["projections"][concept] = {
            "original": {"mean": float(orig_proj.mean()), "std": float(orig_proj.std())},
            "rewritten": {"mean": float(rew_proj.mean()), "std": float(rew_proj.std())},
            "delta_mean": float(rew_proj.mean() - orig_proj.mean()),
            "pct_change": float((rew_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100),
        }
        print(f"  '{concept[:35]}':")
        print(f"    Original: mean|proj|={orig_proj.mean():.4f} ± {orig_proj.std():.4f}")
        print(f"    Rewritten: mean|proj|={rew_proj.mean():.4f} ± {rew_proj.std():.4f}")
        print(f"    Change: {rew_proj.mean() - orig_proj.mean():+.4f} "
              f"({(rew_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100:+.1f}%)")

    # ── Compare cosine similarity between original and rewritten ──
    cos_sims = np.sum(orig_embs * rew_embs, axis=1)
    results["rewrite_similarity"] = {
        "mean_cosine": float(cos_sims.mean()),
        "std_cosine": float(cos_sims.std()),
        "min_cosine": float(cos_sims.min()),
    }
    print(f"\n  Orig↔Rewritten cosine: mean={cos_sims.mean():.4f} "
          f"min={cos_sims.min():.4f} std={cos_sims.std():.4f}")

    # ── Retrieval evaluation ──
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL EVALUATION (full corpus)")
    print(f"{'='*60}")

    per_query_eval = RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    def evaluate_retrieval(q_embs, c_embs, label):
        sims = q_embs @ c_embs.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {all_doc_ids[idx]: float(sims[i, idx]) for idx in top}
        pq = per_query_eval.evaluate(res)
        scores = [pq[qid]["ndcg_cut_10"] for qid in query_ids if qid in pq]
        return float(np.mean(scores)), pq, res

    # Baseline (original corpus)
    orig_ndcg, orig_pq, _ = evaluate_retrieval(query_embs, full_orig_embs, "original")
    rew_ndcg, rew_pq, _ = evaluate_retrieval(query_embs, full_rew_embs, "rewritten")

    results["retrieval"] = {
        "original_corpus_ndcg": orig_ndcg,
        "rewritten_corpus_ndcg": rew_ndcg,
        "delta": round(rew_ndcg - orig_ndcg, 4),
    }
    print(f"\n  Baseline: Original={orig_ndcg:.4f}  Rewritten={rew_ndcg:.4f}  "
          f"Δ={rew_ndcg - orig_ndcg:+.4f}")

    # Subtraction on both corpora
    results["subtraction"] = {}
    for concept in SUBTRACTION_CONCEPTS:
        concept_emb = model.encode(concept, normalize_embeddings=True)
        sub_embs = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])

        orig_sub_ndcg, _, _ = evaluate_retrieval(sub_embs, full_orig_embs, f"orig-sub-{concept}")
        rew_sub_ndcg, _, _ = evaluate_retrieval(sub_embs, full_rew_embs, f"rew-sub-{concept}")

        orig_delta = orig_sub_ndcg - orig_ndcg
        rew_delta = rew_sub_ndcg - rew_ndcg

        results["subtraction"][concept] = {
            "original_delta": round(orig_delta, 4),
            "rewritten_delta": round(rew_delta, 4),
            "degradation_reduced": round(rew_delta - orig_delta, 4),
        }
        print(f"  Sub '{concept[:30]}':")
        print(f"    Original: Δ={orig_delta:+.4f}  Rewritten: Δ={rew_delta:+.4f}  "
              f"Degradation change: {rew_delta - orig_delta:+.4f}")

    # Rotation on both corpora
    target_emb = model.encode(ROTATION_TARGET, normalize_embeddings=True)
    results["rotation"] = {}
    for alpha in [0.1, 0.2]:
        rot_embs = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
        orig_rot_ndcg, _, _ = evaluate_retrieval(rot_embs, full_orig_embs, f"orig-rot-{alpha}")
        rew_rot_ndcg, _, _ = evaluate_retrieval(rot_embs, full_rew_embs, f"rew-rot-{alpha}")

        results["rotation"][f"alpha={alpha}"] = {
            "original_ndcg": round(orig_rot_ndcg, 4),
            "rewritten_ndcg": round(rew_rot_ndcg, 4),
            "original_delta": round(orig_rot_ndcg - orig_ndcg, 4),
            "rewritten_delta": round(rew_rot_ndcg - rew_ndcg, 4),
        }
        print(f"  Rot α={alpha}:")
        print(f"    Original: {orig_rot_ndcg:.4f} (Δ={orig_rot_ndcg-orig_ndcg:+.4f})  "
              f"Rewritten: {rew_rot_ndcg:.4f} (Δ={rew_rot_ndcg-rew_ndcg:+.4f})")

    # Save
    with open(RESULTS_DIR / "semantic_preprocess_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — Item 10")
    print(f"{'='*60}")
    print(f"  Hypothesis: semantic preprocessing reduces projection magnitudes")
    proj_changes = [v["pct_change"] for v in results["projections"].values()]
    print(f"  Mean projection change: {np.mean(proj_changes):+.1f}%")
    print(f"  Hypothesis: subtraction degrades less on rewritten corpus")
    for concept, sdata in results["subtraction"].items():
        print(f"    '{concept[:30]}': degradation change = {sdata['degradation_reduced']:+.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
