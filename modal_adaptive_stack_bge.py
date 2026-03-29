"""
STEER — bge-base Adaptive Stack Completion

Completes the adaptive stack experiment for BAAI/bge-base-en-v1.5
which OOM'd on L4 in the original run. Uses A10G with more memory
and processes datasets sequentially with gc.collect() between them.

Usage: modal run modal_adaptive_stack_bge.py
Author: Renato Aparecido Gomes
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0", "beir", "torch", "numpy",
        "scipy", "pytrec_eval", "datasets", "faiss-cpu", "scikit-learn",
        "transformers>=4.40", "accelerate",
    )
)

app = modal.App("steer-bge-base-stack", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

DATASETS_CONFIG = {
    "scifact": {"target": "clinical medicine and patient outcomes", "domain": "biomedical research"},
    "arguana": {"target": "legal reasoning and jurisprudence", "domain": "argumentation and debate"},
    "nfcorpus": {"target": "clinical nutrition interventions", "domain": "nutrition and health"},
    "fiqa": {"target": "macroeconomic policy impacts", "domain": "financial QA"},
    "trec-covid": {"target": "COVID-19 clinical treatment protocols", "domain": "COVID-19 research"},
}

ALPHA_MAX = 0.2
RRF_K = 60
MAX_QUERIES = 75  # Reduced from 100 to save memory


@app.function(gpu="A10G", memory=49152, timeout=7200, volumes={"/results": vol})
def run_bge_base():
    """Run adaptive stack for bge-base only, with memory management."""
    import gc
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "BAAI/bge-base-en-v1.5"
    print(f"\n{'='*60}")
    print(f"  bge-base Adaptive Stack (A10G, MAX_Q={MAX_QUERIES})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    print("  Loading Qwen2.5-3B-Instruct...")
    llm_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="auto")

    def generate_targets(query, domain, n=3):
        prompt = f"""Given a query from {domain}, suggest {n} SHORT phrases (3-6 words) of adjacent domains with relevant insights.
Query: {query[:300]}
Return ONLY {n} phrases, one per line:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        targets = [l.strip().strip("-•·").strip() for l in resp.split("\n") if l.strip() and len(l.strip()) > 3]
        return targets[:n]

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def remove_top_k(embs, k=1):
        pca = PCA()
        pca.fit(embs)
        result = embs.copy()
        comps = pca.components_[:k]
        for c in comps:
            result -= (result @ c).reshape(-1, 1) * c
        return normalize_rows(result), comps

    def rrf_fusion(scores_list, k=60):
        n = scores_list[0].shape[0]
        rrf = np.zeros(n)
        for s in scores_list:
            ranks = np.argsort(np.argsort(-s)) + 1
            rrf += 1.0 / (k + ranks)
        return rrf

    def eval_ndcg(results_dict, qrels):
        ndcg, _, _, _ = evaluator.evaluate(qrels, results_dict, [1, 5, 10])
        return ndcg

    def scores_to_results(scores, query_ids, doc_ids):
        results = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(scores[i])[::-1][:100]
            results[qid] = {doc_ids[idx]: float(scores[i, idx]) for idx in top}
        return results

    model_results = {"model": model_name, "family": "contrastive"}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
        print(f"\n  Dataset: {ds_name}")
        gc.collect()
        torch.cuda.empty_cache()

        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        except Exception as e:
            print(f"    ERROR: {e}")
            model_results[ds_name] = {"error": str(e)}
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())[:MAX_QUERIES]
        query_texts = [queries[q] for q in query_ids]
        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True))
        generic_target = embed_model.encode(ds_cfg["target"], normalize_embeddings=True)

        corpus_tk, comps = remove_top_k(corpus_embs, k=1)
        query_tk = query_embs.copy()
        for c in comps:
            query_tk -= (query_tk @ c).reshape(-1, 1) * c
        query_tk = normalize_rows(query_tk)
        generic_tk = generic_target.copy()
        for c in comps:
            generic_tk -= np.dot(generic_tk, c) * c
        generic_tk = normalize_vec(generic_tk)

        print(f"    Generating per-query targets...")
        pq_targets = {}
        for idx, (qid, qt) in enumerate(zip(query_ids, query_texts)):
            pq_targets[qid] = generate_targets(qt, ds_cfg["domain"])
            if (idx + 1) % 25 == 0:
                print(f"      {idx+1}/{len(query_ids)} done")

        unique_tgts = list(set(t for ts in pq_targets.values() for t in ts))
        tgt_embs = {}
        if unique_tgts:
            embs = np.array(embed_model.encode(unique_tgts, normalize_embeddings=True))
            tgt_embs = dict(zip(unique_tgts, embs))

        sims_base = query_embs @ corpus_embs.T
        sims_base_tk = query_tk @ corpus_tk.T

        ndcg_a = eval_ndcg(scores_to_results(sims_base, query_ids, doc_ids), qrels)
        base_10 = ndcg_a.get("NDCG@10", 0)

        # B: Adaptive alpha only
        b_scores = np.zeros_like(sims_base)
        for i, qid in enumerate(query_ids):
            best = sims_base[i].copy()
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text]
                sim = float(query_embs[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_embs[i] + alpha * t)
                best = np.maximum(best, q_rot @ corpus_embs.T)
            b_scores[i] = best
        ndcg_b = eval_ndcg(scores_to_results(b_scores, query_ids, doc_ids), qrels)

        # C: MV generic
        q_add = normalize_rows(query_embs + ALPHA_MAX * generic_target)
        sims_add = q_add @ corpus_embs.T
        c_res = {}
        for i, qid in enumerate(query_ids):
            rrf = rrf_fusion([sims_base[i], sims_add[i]], k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            c_res[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_c = eval_ndcg(c_res, qrels)

        # D: Adapt + MV no top-k
        d_res = {}
        for i, qid in enumerate(query_ids):
            sl = [sims_base[i]]
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text]
                sim = float(query_embs[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_embs[i] + alpha * t)
                sl.append(q_rot @ corpus_embs.T)
            rrf = rrf_fusion(sl, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            d_res[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_d = eval_ndcg(d_res, qrels)

        # E: Full stack
        e_res = {}
        for i, qid in enumerate(query_ids):
            sl = [sims_base_tk[i]]
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text].copy()
                for c in comps:
                    t -= np.dot(t, c) * c
                t = normalize_vec(t)
                sim = float(query_tk[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_tk[i] + alpha * t)
                sl.append(q_rot @ corpus_tk.T)
            rrf = rrf_fusion(sl, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            e_res[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_e = eval_ndcg(e_res, qrels)

        model_results[ds_name] = {
            "A_baseline": {k: round(v, 4) for k, v in ndcg_a.items()},
            "B_adaptive_alpha_only": {"ndcg": {k: round(v, 4) for k, v in ndcg_b.items()}, "delta": round(ndcg_b.get("NDCG@10", 0) - base_10, 4)},
            "C_multivector_generic": {"ndcg": {k: round(v, 4) for k, v in ndcg_c.items()}, "delta": round(ndcg_c.get("NDCG@10", 0) - base_10, 4)},
            "D_adapt_mv_no_topk": {"ndcg": {k: round(v, 4) for k, v in ndcg_d.items()}, "delta": round(ndcg_d.get("NDCG@10", 0) - base_10, 4)},
            "E_full_stack": {"ndcg": {k: round(v, 4) for k, v in ndcg_e.items()}, "delta": round(ndcg_e.get("NDCG@10", 0) - base_10, 4)},
        }
        print(f"    A={base_10:.4f} B={ndcg_b.get('NDCG@10',0)-base_10:+.4f} C={ndcg_c.get('NDCG@10',0)-base_10:+.4f} D={ndcg_d.get('NDCG@10',0)-base_10:+.4f} E={ndcg_e.get('NDCG@10',0)-base_10:+.4f}")

        # Cleanup
        del corpus_embs, query_embs, corpus_tk, query_tk, sims_base, sims_base_tk
        gc.collect()
        torch.cuda.empty_cache()

    out_path = "/results/adaptive_stack/BAAI_bge-base-en-v1.5.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — bge-base Adaptive Stack (A10G, 5 datasets)")
    print("=" * 70)
    result = run_bge_base.remote()
    print("\n  DONE!")
