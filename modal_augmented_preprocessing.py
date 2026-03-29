"""
A²RAG — Augmented Preprocessing (NOT destructive rewriting).

Key insight: AUGMENT documents by adding semantic glosses to technical terms,
WITHOUT removing the original terms. This preserves discriminative information
while adding semantic bridges for cross-domain rotation.

Previous approach (failed): "Replace ALL jargon with plain language" → destroyed info
New approach: "Add parenthetical explanations to jargon, keep originals"

Usage: modal run modal_augmented_preprocessing.py

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

app = modal.App("a2rag-augmented-preproc", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

# The critical difference: AUGMENT, don't REPLACE
AUGMENT_PROMPT = """Add brief parenthetical explanations to technical terms, abbreviations, formulas, and jargon in this scientific abstract. KEEP all original terms — only ADD clarifications in parentheses after them. Do not remove or replace any original text. Be concise.

Example:
Input: "BRCA1 mutation in p53-deficient cells shows increased apoptosis (p < 0.01)"
Output: "BRCA1 (DNA repair gene) mutation in p53-deficient (lacking tumor suppressor) cells shows increased apoptosis (programmed cell death) (p < 0.01, statistically significant)"

Now augment this abstract:

{text}

Augmented version:"""

# For comparison: the destructive approach
DESTRUCTIVE_PROMPT = """Rewrite this scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use everyday language. Be concise — output only the rewritten text, nothing else.

Original:
{text}

Rewritten:"""


@app.function(gpu="A10G", memory=32768, timeout=5400, volumes={"/results": vol})
def run_augmented_preprocessing():
    """Compare augmented vs destructive preprocessing on SciFact."""
    import numpy as np
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print("=" * 70)
    print("  Augmented vs Destructive Preprocessing")
    print("=" * 70)

    # Load SciFact
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Select 100 docs
    np.random.seed(42)
    relevant_docs = set()
    for qid, rels in qrels.items():
        for did, score in rels.items():
            if score > 0:
                relevant_docs.add(did)
    selected_idx = [i for i, did in enumerate(doc_ids) if did in relevant_docs][:100]
    remaining = [i for i in range(len(doc_ids)) if i not in selected_idx]
    np.random.shuffle(remaining)
    selected_idx.extend(remaining[:max(0, 100 - len(selected_idx))])
    selected_idx = sorted(selected_idx[:100])

    # Load LLM
    print("  Loading Qwen2.5-7B-Instruct...")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    gen = pipeline("text-generation", model=llm, tokenizer=tokenizer,
                   max_new_tokens=600, temperature=0.3, do_sample=True,
                   return_full_text=False)

    # Process both approaches
    augmented_texts = {}
    destructive_texts = {}

    for count, idx in enumerate(selected_idx):
        did = doc_ids[idx]
        original = doc_texts[idx][:1500]

        # Augmented (preserve + add glosses)
        try:
            out = gen(AUGMENT_PROMPT.format(text=original))
            augmented = out[0]["generated_text"].strip()
            if "\n\n" in augmented:
                augmented = augmented.split("\n\n")[0].strip()
            augmented_texts[did] = augmented if len(augmented) > len(original) * 0.5 else original
        except Exception as e:
            augmented_texts[did] = original

        # Destructive (replace jargon)
        try:
            out = gen(DESTRUCTIVE_PROMPT.format(text=original))
            destructive = out[0]["generated_text"].strip()
            if "\n\n" in destructive:
                destructive = destructive.split("\n\n")[0].strip()
            destructive_texts[did] = destructive if len(destructive) > 20 else original
        except Exception as e:
            destructive_texts[did] = original

        if (count + 1) % 10 == 0:
            print(f"  [{count+1}/100] processed")
            # Save checkpoints
            os.makedirs("/results/augmented_preprocessing", exist_ok=True)
            with open("/results/augmented_preprocessing/augmented_texts.json", "w") as f:
                json.dump(augmented_texts, f, indent=2)
            with open("/results/augmented_preprocessing/destructive_texts.json", "w") as f:
                json.dump(destructive_texts, f, indent=2)
            vol.commit()

    # Save final texts
    os.makedirs("/results/augmented_preprocessing", exist_ok=True)
    with open("/results/augmented_preprocessing/augmented_texts.json", "w") as f:
        json.dump(augmented_texts, f, indent=2)
    with open("/results/augmented_preprocessing/destructive_texts.json", "w") as f:
        json.dump(destructive_texts, f, indent=2)

    # Free LLM
    del llm, gen, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Evaluate with BGE-small ──
    print("\n  Evaluating with BGE-small...")
    eval_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    evaluator = EvaluateRetrieval()

    # Encode everything
    all_corpus_embs = np.array(eval_model.encode(doc_texts, batch_size=256,
                                                  normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(eval_model.encode(query_texts, normalize_embeddings=True,
                                             show_progress_bar=False))

    rewritten_idx = [i for i, did in enumerate(doc_ids) if did in augmented_texts]

    orig_subset = [doc_texts[i] for i in rewritten_idx]
    aug_subset = [augmented_texts.get(doc_ids[i], doc_texts[i]) for i in rewritten_idx]
    dest_subset = [destructive_texts.get(doc_ids[i], doc_texts[i]) for i in rewritten_idx]

    orig_embs = np.array(eval_model.encode(orig_subset, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=False))
    aug_embs = np.array(eval_model.encode(aug_subset, batch_size=256,
                                           normalize_embeddings=True, show_progress_bar=False))
    dest_embs = np.array(eval_model.encode(dest_subset, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=False))

    # Build hybrid corpora
    aug_full = all_corpus_embs.copy()
    dest_full = all_corpus_embs.copy()
    for i, idx in enumerate(rewritten_idx):
        aug_full[idx] = aug_embs[i]
        dest_full[idx] = dest_embs[i]

    def eval_ndcg(q, c):
        sims = q @ c.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
        return ndcg.get("NDCG@10", 0)

    # Similarity analysis
    aug_cos = np.sum(orig_embs * aug_embs, axis=1)
    dest_cos = np.sum(orig_embs * dest_embs, axis=1)

    # Projection analysis
    concepts = {
        "methodology": "methodology and statistical analysis",
        "animal_studies": "animal model studies",
        "genetics": "genetic analysis",
        "clinical": "clinical medicine and patient outcomes",
    }

    proj_results = {}
    for cname, ctext in concepts.items():
        cemb = eval_model.encode(ctext, normalize_embeddings=True)
        op = float(np.abs(orig_embs @ cemb).mean())
        ap = float(np.abs(aug_embs @ cemb).mean())
        dp = float(np.abs(dest_embs @ cemb).mean())
        proj_results[cname] = {
            "original": op,
            "augmented": ap, "aug_delta_pct": round((ap - op) / op * 100, 1),
            "destructive": dp, "dest_delta_pct": round((dp - op) / op * 100, 1),
        }
        print(f"  Proj '{cname}': orig={op:.4f} aug={ap:.4f} ({(ap-op)/op*100:+.1f}%) dest={dp:.4f} ({(dp-op)/op*100:+.1f}%)")

    # Retrieval
    baseline = eval_ndcg(query_embs, all_corpus_embs)
    aug_ndcg = eval_ndcg(query_embs, aug_full)
    dest_ndcg = eval_ndcg(query_embs, dest_full)

    # Rotation on each
    target_emb = eval_model.encode("clinical medicine and patient outcomes", normalize_embeddings=True)
    rot_q = np.array([((q + 0.1 * target_emb) / np.linalg.norm(q + 0.1 * target_emb))
                       for q in query_embs])  # Using ADDITION (the better operation)

    rot_base = eval_ndcg(rot_q, all_corpus_embs)
    rot_aug = eval_ndcg(rot_q, aug_full)
    rot_dest = eval_ndcg(rot_q, dest_full)

    # Isotropy
    np.random.seed(42)
    n = len(rewritten_idx)
    ia = np.random.randint(0, n, 3000)
    ib = np.random.randint(0, n, 3000)
    mask = ia != ib
    ia, ib = ia[mask], ib[mask]
    orig_iso = float(np.sum(orig_embs[ia] * orig_embs[ib], axis=1).mean())
    aug_iso = float(np.sum(aug_embs[ia] * aug_embs[ib], axis=1).mean())
    dest_iso = float(np.sum(dest_embs[ia] * dest_embs[ib], axis=1).mean())

    # Length analysis
    orig_lengths = [len(doc_texts[i]) for i in rewritten_idx]
    aug_lengths = [len(aug_subset[j]) for j in range(len(rewritten_idx))]
    dest_lengths = [len(dest_subset[j]) for j in range(len(rewritten_idx))]

    results = {
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "eval_model": "BAAI/bge-small-en-v1.5",
            "n_docs": len(rewritten_idx),
        },
        "similarity": {
            "augmented_mean": float(aug_cos.mean()),
            "augmented_std": float(aug_cos.std()),
            "destructive_mean": float(dest_cos.mean()),
            "destructive_std": float(dest_cos.std()),
        },
        "lengths": {
            "original_mean": float(np.mean(orig_lengths)),
            "augmented_mean": float(np.mean(aug_lengths)),
            "destructive_mean": float(np.mean(dest_lengths)),
            "augmented_ratio": float(np.mean(aug_lengths) / np.mean(orig_lengths)),
            "destructive_ratio": float(np.mean(dest_lengths) / np.mean(orig_lengths)),
        },
        "projections": proj_results,
        "retrieval": {
            "baseline": baseline,
            "augmented": aug_ndcg, "aug_delta": round(aug_ndcg - baseline, 4),
            "destructive": dest_ndcg, "dest_delta": round(dest_ndcg - baseline, 4),
        },
        "rotation_addition_0.1": {
            "on_baseline": rot_base, "rot_delta_base": round(rot_base - baseline, 4),
            "on_augmented": rot_aug, "rot_delta_aug": round(rot_aug - baseline, 4),
            "on_destructive": rot_dest, "rot_delta_dest": round(rot_dest - baseline, 4),
        },
        "isotropy": {
            "original": orig_iso,
            "augmented": aug_iso,
            "destructive": dest_iso,
        },
    }

    print(f"\n  {'='*60}")
    print(f"  RESULTS COMPARISON")
    print(f"  {'='*60}")
    print(f"  Similarity:  aug={aug_cos.mean():.4f}  dest={dest_cos.mean():.4f}")
    print(f"  Length ratio: aug={np.mean(aug_lengths)/np.mean(orig_lengths):.2f}x  dest={np.mean(dest_lengths)/np.mean(orig_lengths):.2f}x")
    print(f"  Baseline nDCG:     {baseline:.4f}")
    print(f"  Augmented nDCG:    {aug_ndcg:.4f} ({aug_ndcg-baseline:+.4f})")
    print(f"  Destructive nDCG:  {dest_ndcg:.4f} ({dest_ndcg-baseline:+.4f})")
    print(f"  Rot+Base:     {rot_base:.4f}")
    print(f"  Rot+Aug:      {rot_aug:.4f}")
    print(f"  Rot+Dest:     {rot_dest:.4f}")
    print(f"  Isotropy: orig={orig_iso:.4f} aug={aug_iso:.4f} dest={dest_iso:.4f}")

    with open("/results/augmented_preprocessing/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"\n  Saved to /results/augmented_preprocessing/")
    return results


@app.local_entrypoint()
def main():
    results = run_augmented_preprocessing.remote()
    print(f"\n  FINAL: Aug Δ={results['retrieval']['aug_delta']:+.4f}  Dest Δ={results['retrieval']['dest_delta']:+.4f}")
