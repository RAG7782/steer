"""
Item 10: Full pipeline on Modal — rewriting with HF Transformers + evaluation.
Single function design (no local entrypoint coordination) for --detach safety.

Usage: modal run --detach modal_item10_full.py

Author: Renato Aparecido Gomes
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0",
        "beir",
        "torch",
        "numpy",
        "scipy",
        "pytrec_eval",
        "datasets",
        "faiss-cpu",
        "transformers>=4.40",
        "accelerate",
    )
)

app = modal.App("a2rag-item10", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

REWRITE_PROMPT = """Rewrite this scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use everyday language. Be concise — output only the rewritten text, nothing else.

Original:
{text}

Rewritten:"""


@app.function(gpu="L4", memory=32768, timeout=5400, volumes={"/results": vol})
def item10_full_pipeline():
    """Single-function pipeline: rewrite 100 docs + evaluate retrieval impact.

    Uses Phi-3-mini-4k-instruct (3.8B params) for rewriting — fits in T4 16GB
    with float16. Then evaluates with BGE-small.
    """
    import numpy as np
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    # ═══ STEP 1: Load SciFact ═══
    print("=" * 70)
    print("  ITEM 10: Semantic Preprocessing Pipeline")
    print("=" * 70)
    print("\n  Loading SciFact...")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Select 100 docs (prioritize those with relevance judgments)
    np.random.seed(42)
    relevant_docs = set()
    for qid, rels in qrels.items():
        for did, score in rels.items():
            if score > 0:
                relevant_docs.add(did)

    selected_idx = []
    for i, did in enumerate(doc_ids):
        if did in relevant_docs and len(selected_idx) < 100:
            selected_idx.append(i)
    remaining = [i for i in range(len(doc_ids)) if i not in selected_idx]
    np.random.shuffle(remaining)
    selected_idx.extend(remaining[:max(0, 100 - len(selected_idx))])
    selected_idx = sorted(selected_idx[:100])

    n_relevant = sum(1 for i in selected_idx if doc_ids[i] in relevant_docs)
    print(f"  Selected {len(selected_idx)} docs ({n_relevant} with relevance)")

    # ═══ STEP 2: Rewrite with Qwen2.5-1.5B-Instruct ═══
    # Using a 1.5B model that fits easily on T4 with plenty of room for data
    print("\n  Loading Qwen2.5-1.5B-Instruct for rewriting...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                   max_new_tokens=400, temperature=0.3, do_sample=True,
                   return_full_text=False)

    rewritten_texts = {}
    for count, idx in enumerate(selected_idx):
        did = doc_ids[idx]
        original = doc_texts[idx]
        prompt = REWRITE_PROMPT.format(text=original[:1500])

        try:
            out = gen(prompt)
            rewritten = out[0]["generated_text"].strip()
            if "\n\n" in rewritten:
                rewritten = rewritten.split("\n\n")[0].strip()
            if len(rewritten) > 20:
                rewritten_texts[did] = rewritten
            else:
                rewritten_texts[did] = original
        except Exception as e:
            print(f"    Failed doc {did}: {e}")
            rewritten_texts[did] = original

        if (count + 1) % 10 == 0:
            print(f"    [{count+1}/{len(selected_idx)}] rewritten")
            # Save incremental checkpoint
            os.makedirs("/results/item10", exist_ok=True)
            with open("/results/item10/rewritten_docs.json", "w") as f:
                json.dump(rewritten_texts, f, indent=2)
            vol.commit()

    # Save final rewritten docs
    os.makedirs("/results/item10", exist_ok=True)
    with open("/results/item10/rewritten_docs.json", "w") as f:
        json.dump(rewritten_texts, f, indent=2)
    vol.commit()
    print(f"\n  Rewrote {len(rewritten_texts)} docs successfully")

    # Free LLM memory
    del model, gen, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══ STEP 3: Evaluate with BGE-small ═══
    print("\n  Loading BGE-small for evaluation...")
    eval_model_name = "BAAI/bge-small-en-v1.5"
    eval_model = SentenceTransformer(eval_model_name)

    rewritten_idx = [i for i, did in enumerate(doc_ids) if did in rewritten_texts]
    print(f"  {len(rewritten_idx)} rewritten docs found in corpus")

    print("  Encoding corpus + queries...")
    all_corpus_embs = np.array(eval_model.encode(doc_texts, batch_size=256,
                                                  normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(eval_model.encode(query_texts, normalize_embeddings=True,
                                             show_progress_bar=False))

    orig_subset = [doc_texts[i] for i in rewritten_idx]
    rewr_subset = [rewritten_texts.get(doc_ids[i], doc_texts[i]) for i in rewritten_idx]

    orig_embs = np.array(eval_model.encode(orig_subset, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=False))
    rewr_embs = np.array(eval_model.encode(rewr_subset, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=False))

    # Build hybrid corpus (replace rewritten docs)
    rewr_full_embs = all_corpus_embs.copy()
    for i, idx in enumerate(rewritten_idx):
        rewr_full_embs[idx] = rewr_embs[i]

    results = {
        "config": {"eval_model": eval_model_name,
                    "rewrite_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "n_docs_rewritten": len(rewritten_idx),
                    "total_docs": len(doc_ids)},
    }

    # Embedding similarity
    cos_sims = np.sum(orig_embs * rewr_embs, axis=1)
    results["embedding_similarity"] = {
        "mean_cosine": float(cos_sims.mean()),
        "std_cosine": float(cos_sims.std()),
        "min_cosine": float(cos_sims.min()),
        "max_cosine": float(cos_sims.max()),
    }
    print(f"  Embedding similarity: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")

    # Projection analysis
    concepts = {
        "methodology": "methodology and statistical analysis",
        "animal_studies": "animal model studies",
        "genetics": "genetic analysis",
        "rotation_target": "clinical medicine and patient outcomes",
    }
    results["projections"] = {}
    for cname, ctext in concepts.items():
        cemb = eval_model.encode(ctext, normalize_embeddings=True)
        orig_proj = np.abs(orig_embs @ cemb)
        rewr_proj = np.abs(rewr_embs @ cemb)
        results["projections"][cname] = {
            "concept": ctext,
            "original": {"mean": float(orig_proj.mean()), "std": float(orig_proj.std())},
            "rewritten": {"mean": float(rewr_proj.mean()), "std": float(rewr_proj.std())},
            "delta_pct": float((rewr_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100),
        }
        print(f"  Proj '{cname}': {orig_proj.mean():.4f} → {rewr_proj.mean():.4f} "
              f"({(rewr_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100:+.1f}%)")

    # Retrieval evaluation
    evaluator = EvaluateRetrieval()

    def eval_ndcg(q_embs, c_embs):
        sims = q_embs @ c_embs.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
        return ndcg.get("NDCG@10", 0)

    def rotate_toward(source_emb, target_emb, alpha=0.4):
        result = (1 - alpha) * source_emb + alpha * target_emb
        norm = np.linalg.norm(result)
        return result / norm if norm > 1e-10 else result

    baseline_ndcg = eval_ndcg(query_embs, all_corpus_embs)
    hybrid_ndcg = eval_ndcg(query_embs, rewr_full_embs)
    results["retrieval"] = {
        "baseline_ndcg10": baseline_ndcg,
        "hybrid_ndcg10": hybrid_ndcg,
        "delta": round(hybrid_ndcg - baseline_ndcg, 4),
    }
    print(f"\n  Baseline nDCG@10: {baseline_ndcg:.4f}")
    print(f"  Hybrid nDCG@10:   {hybrid_ndcg:.4f} ({hybrid_ndcg - baseline_ndcg:+.4f})")

    # Rotation on hybrid vs original
    target_emb = eval_model.encode("clinical medicine and patient outcomes",
                                    normalize_embeddings=True)
    rotated_q = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])
    rot_orig = eval_ndcg(rotated_q, all_corpus_embs)
    rot_hybrid = eval_ndcg(rotated_q, rewr_full_embs)
    results["rotation_0.1"] = {
        "on_original": rot_orig,
        "on_hybrid": rot_hybrid,
        "delta_orig_vs_base": round(rot_orig - baseline_ndcg, 4),
        "delta_hybrid_vs_base": round(rot_hybrid - baseline_ndcg, 4),
    }
    print(f"  Rotation α=0.1: orig={rot_orig:.4f} hybrid={rot_hybrid:.4f}")

    # Isotropy comparison
    np.random.seed(42)
    n = len(rewritten_idx)
    if n > 1:
        ia = np.random.randint(0, n, 3000)
        ib = np.random.randint(0, n, 3000)
        mask = ia != ib
        ia, ib = ia[mask], ib[mask]
        orig_iso = float(np.sum(orig_embs[ia] * orig_embs[ib], axis=1).mean())
        rewr_iso = float(np.sum(rewr_embs[ia] * rewr_embs[ib], axis=1).mean())
        results["isotropy"] = {"original": orig_iso, "rewritten": rewr_iso}
        print(f"  Isotropy: orig={orig_iso:.4f} → rewr={rewr_iso:.4f}")

    # ═══ SAVE FINAL RESULTS ═══
    with open("/results/item10/preprocessing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"\n  ALL DONE! Saved to /results/item10/")

    return results


@app.local_entrypoint()
def main():
    """Calls the single remote function."""
    results = item10_full_pipeline.remote()
    print(f"\n{'='*70}")
    print(f"  ITEM 10 COMPLETE")
    print(f"{'='*70}")
    print(f"  Similarity: {results['embedding_similarity']['mean_cosine']:.4f}")
    print(f"  Baseline:   {results['retrieval']['baseline_ndcg10']:.4f}")
    print(f"  Hybrid:     {results['retrieval']['hybrid_ndcg10']:.4f} "
          f"({results['retrieval']['delta']:+.4f})")
    print(f"\n  Download: modal volume get a2rag-results item10/ ./results_modal/item10/")
