"""
A²RAG — Tests for Future Work items 6, 10, 12, 15 (now numbered in updated paper).

Item 6 (FW): Isotropy-aware model selection — test whitening/PCA on high-isotropy models
Item 10 (FW): Semantic preprocessing — use larger LLM (gemma-2-9b-it) for rewriting
Item 12 (FW/Appendix): Boolean Conceptors — comprehensive test with tuned apertures
Item 15 (FW/Appendix): Vector Addition — systematic comparison with NLERP

Usage: modal run --detach modal_future_work_tests.py

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
        "scikit-learn",
    )
)

image_llm = (
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

app = modal.App("a2rag-future-work", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

DATASETS = ["scifact", "arguana"]


def op_nlerp(query_embs, target_emb, alpha=0.1):
    import numpy as np
    results = (1 - alpha) * query_embs + alpha * target_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_addition(query_embs, concept_emb, alpha=0.1):
    import numpy as np
    results = query_embs + alpha * concept_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_subtraction(query_embs, exclude_emb):
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


# ═══════════════════════════════════════════════════════════════════
# FW ITEM 6: Isotropy-aware — test whitening on high-isotropy models
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def fw6_isotropy_whitening(model_name: str):
    """Test if PCA whitening makes high-isotropy models amenable to rotation."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"FW6: Whitening test for {model_name}")
    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    all_results = {"model": model_name}

    for ds_name in DATASETS:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))

        if ds_name == "scifact":
            target_text = "clinical medicine and patient outcomes"
        else:
            target_text = "economic policy and market regulation"
        target_emb = model.encode(target_text, normalize_embeddings=True)

        def eval_ndcg(q, c):
            sims = q @ c.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        def measure_isotropy(embs):
            np.random.seed(42)
            n = min(3000, len(embs))
            ia = np.random.randint(0, len(embs), n)
            ib = np.random.randint(0, len(embs), n)
            mask = ia != ib
            return float(np.sum(embs[ia[mask]] * embs[ib[mask]], axis=1).mean())

        # Original
        iso_orig = measure_isotropy(corpus_embs)
        base_orig = eval_ndcg(query_embs, corpus_embs)
        rot_orig = eval_ndcg(op_nlerp(query_embs, target_emb, 0.1), corpus_embs)

        # PCA Whitening (fit on corpus, transform both)
        for n_components in [None, 256, 128]:
            pca = PCA(n_components=n_components, whiten=True)
            corpus_w = pca.fit_transform(corpus_embs)
            query_w = pca.transform(query_embs)
            target_w = pca.transform(target_emb.reshape(1, -1))[0]

            # Re-normalize
            corpus_w = corpus_w / (np.linalg.norm(corpus_w, axis=1, keepdims=True) + 1e-10)
            query_w = query_w / (np.linalg.norm(query_w, axis=1, keepdims=True) + 1e-10)
            target_w = target_w / (np.linalg.norm(target_w) + 1e-10)

            iso_w = measure_isotropy(corpus_w)
            base_w = eval_ndcg(query_w, corpus_w)
            rot_w = eval_ndcg(op_nlerp(query_w, target_w, 0.1), corpus_w)

            dim_label = n_components or corpus_embs.shape[1]
            print(f"  {ds_name} dim={dim_label}: iso {iso_orig:.3f}→{iso_w:.3f}, base {base_orig:.4f}→{base_w:.4f}, rot Δ {rot_orig-base_orig:+.4f}→{rot_w-base_w:+.4f}")

            all_results.setdefault(ds_name, {})[f"whitened_d{dim_label}"] = {
                "isotropy_before": iso_orig, "isotropy_after": iso_w,
                "baseline_before": base_orig, "baseline_after": base_w,
                "rotation_delta_before": round(rot_orig - base_orig, 4),
                "rotation_delta_after": round(rot_w - base_w, 4),
            }

        all_results[ds_name]["original"] = {
            "isotropy": iso_orig, "baseline": base_orig,
            "rotation_delta": round(rot_orig - base_orig, 4),
        }

    safe = model_name.replace("/", "_")
    out_path = f"/results/fw6_whitening/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    return all_results


# ═══════════════════════════════════════════════════════════════════
# FW ITEM 10: Semantic preprocessing with larger LLM (gemma-2-9b-it)
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="A10G", memory=32768, timeout=5400, volumes={"/results": vol}, image=image_llm)
def fw10_preprocessing_large_llm():
    """Rewrite 100 SciFact docs with Qwen2.5-7B-Instruct (open, not gated)."""
    import numpy as np
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    REWRITE_PROMPT = """Rewrite this scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use everyday language. Be concise — output only the rewritten text, nothing else.

Original:
{text}

Rewritten:"""

    # Load SciFact
    print("FW10: Semantic preprocessing with Qwen/Qwen2.5-7B-Instruct")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

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
                   max_new_tokens=400, temperature=0.3, do_sample=True,
                   return_full_text=False)

    rewritten_texts = {}
    for count, idx in enumerate(selected_idx):
        did = doc_ids[idx]
        original = doc_texts[idx]
        try:
            out = gen(REWRITE_PROMPT.format(text=original[:1500]))
            rewritten = out[0]["generated_text"].strip()
            if "\n\n" in rewritten:
                rewritten = rewritten.split("\n\n")[0].strip()
            if len(rewritten) > 20:
                rewritten_texts[did] = rewritten
            else:
                rewritten_texts[did] = original
        except Exception as e:
            print(f"  Failed {did}: {e}")
            rewritten_texts[did] = original
        if (count + 1) % 10 == 0:
            print(f"  [{count+1}/100] rewritten")
            os.makedirs("/results/fw10_preprocessing", exist_ok=True)
            with open("/results/fw10_preprocessing/rewritten_qwen7b.json", "w") as f:
                json.dump(rewritten_texts, f, indent=2)
            vol.commit()

    os.makedirs("/results/fw10_preprocessing", exist_ok=True)
    with open("/results/fw10_preprocessing/rewritten_qwen7b.json", "w") as f:
        json.dump(rewritten_texts, f, indent=2)

    del llm, gen, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate
    print("  Evaluating with BGE-small...")
    eval_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    evaluator = EvaluateRetrieval()

    rewritten_idx = [i for i, did in enumerate(doc_ids) if did in rewritten_texts]
    all_corpus_embs = np.array(eval_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(eval_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
    orig_subset = [doc_texts[i] for i in rewritten_idx]
    rewr_subset = [rewritten_texts.get(doc_ids[i], doc_texts[i]) for i in rewritten_idx]
    orig_embs = np.array(eval_model.encode(orig_subset, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
    rewr_embs = np.array(eval_model.encode(rewr_subset, batch_size=256, normalize_embeddings=True, show_progress_bar=False))

    rewr_full_embs = all_corpus_embs.copy()
    for i, idx in enumerate(rewritten_idx):
        rewr_full_embs[idx] = rewr_embs[i]

    cos_sims = np.sum(orig_embs * rewr_embs, axis=1)

    def eval_ndcg(q, c):
        sims = q @ c.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
        return ndcg.get("NDCG@10", 0)

    baseline = eval_ndcg(query_embs, all_corpus_embs)
    hybrid = eval_ndcg(query_embs, rewr_full_embs)

    # Projections
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
        rp = float(np.abs(rewr_embs @ cemb).mean())
        proj_results[cname] = {"original": op, "rewritten": rp, "delta_pct": round((rp-op)/op*100, 1)}

    results = {
        "rewrite_model": model_id,
        "n_rewritten": len(rewritten_texts),
        "embedding_similarity": {"mean": float(cos_sims.mean()), "std": float(cos_sims.std())},
        "baseline_ndcg10": baseline,
        "hybrid_ndcg10": hybrid,
        "delta": round(hybrid - baseline, 4),
        "projections": proj_results,
    }
    print(f"  Baseline: {baseline:.4f}, Hybrid: {hybrid:.4f} ({hybrid-baseline:+.4f})")

    with open("/results/fw10_preprocessing/results_qwen7b.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results


# ═══════════════════════════════════════════════════════════════════
# FW ITEM 12: Conceptors — comprehensive with tuned apertures
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def fw12_conceptors_comprehensive():
    """Comprehensive conceptor test with many apertures and concept construction methods."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print("FW12: Comprehensive Boolean Conceptor evaluation")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    evaluator = EvaluateRetrieval()
    all_results = {}

    for ds_name in DATASETS:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))

        if ds_name == "scifact":
            concept_queries = ["clinical trials", "patient treatment", "drug efficacy", "disease outcomes", "medical intervention"]
        else:
            concept_queries = ["tax policy", "market regulation", "fiscal stimulus", "trade agreement", "economic growth"]

        cq_embs = np.array(model.encode(concept_queries, normalize_embeddings=True))
        sims = cq_embs.mean(axis=0) @ corpus_embs.T

        def eval_ndcg(q, c):
            sims_m = q @ c.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims_m[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims_m[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        baseline = eval_ndcg(query_embs, corpus_embs)
        ds_r = {"baseline_ndcg10": baseline}

        # Vary concept corpus size and aperture
        for n_docs in [10, 25, 50, 100, 200]:
            concept_idx = np.argsort(sims)[::-1][:n_docs]
            concept_embs = corpus_embs[concept_idx]

            for aperture in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
                # Build conceptor
                n, d = concept_embs.shape
                R = concept_embs.T @ concept_embs / n
                alpha_sq_inv = 1.0 / (aperture ** 2)
                try:
                    C = R @ np.linalg.inv(R + alpha_sq_inv * np.eye(d))
                except np.linalg.LinAlgError:
                    continue

                # Apply conceptor
                filtered = query_embs @ C.T
                norms = np.linalg.norm(filtered, axis=1, keepdims=True)
                filtered = filtered / np.maximum(norms, 1e-10)
                ndcg = eval_ndcg(filtered, corpus_embs)

                # NOT conceptor
                C_not = np.eye(d) - C
                not_filtered = query_embs @ C_not.T
                norms = np.linalg.norm(not_filtered, axis=1, keepdims=True)
                not_filtered = not_filtered / np.maximum(norms, 1e-10)
                ndcg_not = eval_ndcg(not_filtered, corpus_embs)

                ds_r[f"filter_n{n_docs}_a{aperture}"] = {"ndcg10": ndcg, "delta": round(ndcg - baseline, 4)}
                ds_r[f"not_n{n_docs}_a{aperture}"] = {"ndcg10": ndcg_not, "delta": round(ndcg_not - baseline, 4)}

                print(f"  {ds_name} n={n_docs} a={aperture}: filter={ndcg:.4f} NOT={ndcg_not:.4f}")

        all_results[ds_name] = ds_r

    out_path = "/results/fw12_conceptors/comprehensive.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    return all_results


# ═══════════════════════════════════════════════════════════════════
# FW ITEM 15: Vector Addition systematic comparison with NLERP
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def fw15_addition_vs_nlerp(model_name: str):
    """Systematic comparison: vector addition vs NLERP across alpha values."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"FW15: Addition vs NLERP for {model_name}")
    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    all_results = {"model": model_name}

    for ds_name in DATASETS:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))

        if ds_name == "scifact":
            target_text = "clinical medicine and patient outcomes"
        else:
            target_text = "economic policy and market regulation"
        target_emb = model.encode(target_text, normalize_embeddings=True)

        def eval_ndcg(q, c):
            sims = q @ c.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        def jaccard(q1, q2, c, k=10):
            s1, s2 = q1 @ c.T, q2 @ c.T
            j = []
            for i in range(len(q1)):
                t1 = set(np.argsort(s1[i])[::-1][:k])
                t2 = set(np.argsort(s2[i])[::-1][:k])
                j.append(len(t1&t2)/len(t1|t2) if t1|t2 else 1.0)
            return float(np.mean(j))

        baseline = eval_ndcg(query_embs, corpus_embs)
        ds_r = {"baseline_ndcg10": baseline}

        for alpha in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            nlerp_q = op_nlerp(query_embs, target_emb, alpha)
            add_q = op_addition(query_embs, target_emb, alpha)

            nlerp_ndcg = eval_ndcg(nlerp_q, corpus_embs)
            add_ndcg = eval_ndcg(add_q, corpus_embs)

            # How similar are the two transformations?
            cross_jacc = jaccard(nlerp_q, add_q, corpus_embs)
            # Angular difference between NLERP and Addition results
            dots = np.sum(nlerp_q * add_q, axis=1)
            angle_diff = float(np.degrees(np.arccos(np.clip(dots, -1, 1))).mean())

            ds_r[f"a{alpha}"] = {
                "nlerp_ndcg10": nlerp_ndcg,
                "addition_ndcg10": add_ndcg,
                "nlerp_delta": round(nlerp_ndcg - baseline, 4),
                "addition_delta": round(add_ndcg - baseline, 4),
                "cross_jaccard": cross_jacc,
                "angle_diff_deg": round(angle_diff, 2),
            }
            print(f"  {ds_name} α={alpha}: NLERP={nlerp_ndcg:.4f} Add={add_ndcg:.4f} cross_J={cross_jacc:.3f} angle={angle_diff:.1f}°")

        all_results[ds_name] = ds_r

    safe = model_name.replace("/", "_")
    out_path = f"/results/fw15_addition/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    return all_results


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  Future Work Tests: Items 6, 10, 12, 15")
    print("=" * 70)

    # FW6: Whitening on 3 high-isotropy models (parallel)
    high_iso_models = ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2", "thenlper/gte-small"]
    fw6_results = list(fw6_isotropy_whitening.map(high_iso_models))
    print("\n  FW6: Whitening complete")

    # FW10: Larger LLM preprocessing
    fw10_result = fw10_preprocessing_large_llm.remote()
    print(f"\n  FW10: Preprocessing with gemma-2-9b: Δ={fw10_result['delta']:+.4f}")

    # FW12: Conceptors comprehensive
    fw12_result = fw12_conceptors_comprehensive.remote()
    print("\n  FW12: Conceptors comprehensive complete")

    # FW15: Addition vs NLERP for all 6 models (parallel)
    models_6 = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "all-mpnet-base-v2",
                "BAAI/bge-base-en-v1.5", "intfloat/e5-small-v2", "thenlper/gte-small"]
    fw15_results = list(fw15_addition_vs_nlerp.map(models_6))
    print("\n  FW15: Addition vs NLERP complete")

    print("\n  ALL DONE!")


# Individual entrypoints for direct CLI invocation
@app.local_entrypoint()
def run_fw6():
    """FW6: Whitening tests for 3 high-isotropy models."""
    high_iso = ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2", "thenlper/gte-small"]
    list(fw6_isotropy_whitening.map(high_iso))
    print("FW6 complete")

@app.local_entrypoint()
def run_fw10():
    """FW10: Preprocessing with gemma-2-9b."""
    r = fw10_preprocessing_large_llm.remote()
    print(f"FW10: delta={r['delta']:+.4f}")

@app.local_entrypoint()
def run_fw12():
    """FW12: Comprehensive conceptors."""
    fw12_conceptors_comprehensive.remote()
    print("FW12 complete")

@app.local_entrypoint()
def run_fw15():
    """FW15: Addition vs NLERP for 6 models."""
    models = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "all-mpnet-base-v2",
              "BAAI/bge-base-en-v1.5", "intfloat/e5-small-v2", "thenlper/gte-small"]
    list(fw15_addition_vs_nlerp.map(models))
    print("FW15 complete")
