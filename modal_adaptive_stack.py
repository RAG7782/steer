"""
A2RAG — Dataset-Adaptive Stack

O exp per-query targets mostrou que full stack (top-k + adaptive alpha + multi-vector)
funciona muito bem para scifact (+0.016 bge-base) mas piora arguana (-0.006).

Hipotese: top-k removal prejudica datasets argumentativos (remove componentes
semanticamente relevantes para argumentacao). Para estes datasets, a stack deve
pular o top-k removal.

Setup: testar todas as combinacoes de componentes:
A. Baseline
B. Adaptive alpha only
C. Multi-vector only
D. Adaptive alpha + Multi-vector (sem top-k)
E. Top-k + Adaptive alpha + Multi-vector (full stack)
F. Top-k + Multi-vector (sem adaptive alpha)

Para cada: usar targets per-query (Qwen2.5-3B) e targets genericos.

Expandir para 5 datasets BEIR: scifact, arguana, nfcorpus, fiqa, trec-covid.

Usage: modal run modal_adaptive_stack.py

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
        "transformers>=4.40",
        "accelerate",
    )
)

app = modal.App("a2rag-adaptive-stack", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("thenlper/gte-small", "general"),
]

DATASETS_CONFIG = {
    "scifact": {"target": "clinical medicine and patient outcomes", "domain": "biomedical research"},
    "arguana": {"target": "legal reasoning and jurisprudence", "domain": "argumentation and debate"},
    "nfcorpus": {"target": "clinical nutrition interventions", "domain": "nutrition and health"},
    "fiqa": {"target": "macroeconomic policy impacts", "domain": "financial QA"},
    "trec-covid": {"target": "COVID-19 clinical treatment protocols", "domain": "COVID-19 research"},
}

ALPHA_MAX = 0.2
RRF_K = 60
MAX_QUERIES = 100


@app.function(gpu="L4", memory=32768, timeout=5400, volumes={"/results": vol})
def run_adaptive_stack(model_name: str, family: str):
    """Test all stack combinations across 5 datasets."""
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  Adaptive Stack: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    # Load small LLM for per-query targets
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

    def eval_ndcg(results_dict):
        ndcg, _, _, _ = evaluator.evaluate(qrels, results_dict, [1, 5, 10])
        return ndcg

    def scores_to_results(scores, query_ids, doc_ids):
        results = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(scores[i])[::-1][:100]
            results[qid] = {doc_ids[idx]: float(scores[i, idx]) for idx in top}
        return results

    model_results = {"model": model_name, "family": family}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
        print(f"\n  Dataset: {ds_name}")
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        except Exception as e:
            print(f"    ERROR loading {ds_name}: {e}")
            model_results[ds_name] = {"error": str(e)}
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())[:MAX_QUERIES]
        query_texts = [queries[q] for q in query_ids]

        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        # Encode
        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        generic_target = embed_model.encode(ds_cfg["target"], normalize_embeddings=True)

        # Isotropy-corrected versions
        corpus_tk, comps = remove_top_k(corpus_embs, k=1)
        query_tk = query_embs.copy()
        for c in comps:
            query_tk -= (query_tk @ c).reshape(-1, 1) * c
        query_tk = normalize_rows(query_tk)
        generic_tk = generic_target.copy()
        for c in comps:
            generic_tk -= np.dot(generic_tk, c) * c
        generic_tk = normalize_vec(generic_tk)

        # Generate per-query targets
        print(f"    Generating per-query targets...")
        pq_targets = {}
        for idx, (qid, qt) in enumerate(zip(query_ids, query_texts)):
            pq_targets[qid] = generate_targets(qt, ds_cfg["domain"])
            if (idx + 1) % 25 == 0:
                print(f"      {idx+1}/{len(query_ids)} done")

        unique_tgts = list(set(t for ts in pq_targets.values() for t in ts))
        tgt_embs = {}
        if unique_tgts:
            embs = np.array(embed_model.encode(unique_tgts, normalize_embeddings=True, show_progress_bar=False))
            tgt_embs = dict(zip(unique_tgts, embs))

        # Baseline sims
        sims_base = query_embs @ corpus_embs.T
        sims_base_tk = query_tk @ corpus_tk.T

        # === A. Baseline ===
        ndcg_a = eval_ndcg(scores_to_results(sims_base, query_ids, doc_ids))
        base_10 = ndcg_a.get("NDCG@10", 0)

        # === B. Adaptive alpha only (per-query targets, no multi-vector, no top-k) ===
        b_scores = np.zeros_like(sims_base)
        for i, qid in enumerate(query_ids):
            best_scores = sims_base[i].copy()
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text]
                sim = float(query_embs[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_embs[i] + alpha * t)
                best_scores = np.maximum(best_scores, q_rot @ corpus_embs.T)
            b_scores[i] = best_scores
        ndcg_b = eval_ndcg(scores_to_results(b_scores, query_ids, doc_ids))

        # === C. Multi-vector RRF only (generic target, no adaptive, no top-k) ===
        q_add = normalize_rows(query_embs + ALPHA_MAX * generic_target)
        sims_add = q_add @ corpus_embs.T
        c_results = {}
        for i, qid in enumerate(query_ids):
            rrf = rrf_fusion([sims_base[i], sims_add[i]], k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            c_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_c = eval_ndcg(c_results)

        # === D. Adaptive alpha + Multi-vector (per-query, NO top-k) ===
        d_results = {}
        for i, qid in enumerate(query_ids):
            score_lists = [sims_base[i]]
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text]
                sim = float(query_embs[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_embs[i] + alpha * t)
                score_lists.append(q_rot @ corpus_embs.T)
            rrf = rrf_fusion(score_lists, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            d_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_d = eval_ndcg(d_results)

        # === E. Full stack: top-k + adaptive + multi-vector ===
        e_results = {}
        for i, qid in enumerate(query_ids):
            score_lists = [sims_base_tk[i]]
            for t_text in pq_targets[qid]:
                if t_text not in tgt_embs: continue
                t = tgt_embs[t_text].copy()
                for c in comps:
                    t -= np.dot(t, c) * c
                t = normalize_vec(t)
                sim = float(query_tk[i] @ t)
                alpha = ALPHA_MAX * (1 - sim) ** 2
                q_rot = normalize_vec(query_tk[i] + alpha * t)
                score_lists.append(q_rot @ corpus_tk.T)
            rrf = rrf_fusion(score_lists, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            e_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_e = eval_ndcg(e_results)

        ds_results = {
            "A_baseline": {k: round(v, 4) for k, v in ndcg_a.items()},
            "B_adaptive_alpha_only": {"ndcg": {k: round(v, 4) for k, v in ndcg_b.items()}, "delta": round(ndcg_b.get("NDCG@10", 0) - base_10, 4)},
            "C_multivector_generic": {"ndcg": {k: round(v, 4) for k, v in ndcg_c.items()}, "delta": round(ndcg_c.get("NDCG@10", 0) - base_10, 4)},
            "D_adapt_mv_no_topk": {"ndcg": {k: round(v, 4) for k, v in ndcg_d.items()}, "delta": round(ndcg_d.get("NDCG@10", 0) - base_10, 4)},
            "E_full_stack": {"ndcg": {k: round(v, 4) for k, v in ndcg_e.items()}, "delta": round(ndcg_e.get("NDCG@10", 0) - base_10, 4)},
        }

        # Print summary
        print(f"    A.Baseline={base_10:.4f}"
              f"  B.Adapt={ndcg_b.get('NDCG@10',0)-base_10:+.4f}"
              f"  C.MV_gen={ndcg_c.get('NDCG@10',0)-base_10:+.4f}"
              f"  D.Adapt+MV={ndcg_d.get('NDCG@10',0)-base_10:+.4f}"
              f"  E.Full={ndcg_e.get('NDCG@10',0)-base_10:+.4f}")

        model_results[ds_name] = ds_results

    safe = model_name.replace("/", "_")
    out_path = f"/results/adaptive_stack/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Dataset-Adaptive Stack (5 datasets)")
    print("=" * 70)

    all_results = list(run_adaptive_stack.starmap(
        [(name, fam) for name, fam in MODELS]
    ))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<25} {'DS':<12} {'Base':>7} {'B.Adpt':>7} {'C.MVgn':>7} {'D.A+MV':>7} {'E.Full':>7}")
    print("  " + "-" * 72)

    for r in all_results:
        for ds in DATASETS_CONFIG:
            if ds not in r or "error" in r[ds]:
                continue
            d = r[ds]
            base = d["A_baseline"].get("NDCG@10", 0)
            print(f"  {r['model']:<25} {ds:<12} {base:>7.4f}"
                  f" {d['B_adaptive_alpha_only']['delta']:>+7.4f}"
                  f" {d['C_multivector_generic']['delta']:>+7.4f}"
                  f" {d['D_adapt_mv_no_topk']['delta']:>+7.4f}"
                  f" {d['E_full_stack']['delta']:>+7.4f}")

    print("\n  DONE!")
