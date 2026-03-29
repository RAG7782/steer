"""
A2RAG — LLM-Generated Per-Query Targets + Full Adaptive Stack

O experimento mais impactante: em vez de um target generico para todas as queries,
usar um LLM para gerar 2-3 targets cross-domain ESPECIFICOS por query.

Depois, aplicar a stack adaptativa completa:
1. Top-k removal (k=1) no corpus (isotropy correction)
2. Adaptive alpha quadratic: alpha(q) = alpha_max * (1 - cos_sim(q, target))^2
3. Multi-vector RRF: buscar q + T(q, target1) + T(q, target2) + T(q, target3)

Hipotese: targets per-query resolvem o problema fundamental do exp 7
(rotacao uniforme degrada). Se cada query tem um target relevante,
os docs surfaceados serao genuinamente cross-domain relevantes.

Comparacoes:
A. Baseline (q somente)
B. Addition uniforme (target generico, alpha fixo) — referencia negativa
C. Multi-vector com target generico (exp 6 revisitado)
D. Multi-vector com targets per-query (NOVO)
E. Stack completa: top-k + adaptive alpha + multi-vector per-query (NOVO)

Avaliacao:
- nDCG@10 no dataset original (preservacao do baseline)
- Docs novos surfaceados (diversidade)
- LLM-judge: relevancia dos docs novos (0-2)

Usage: modal run modal_perquery_targets.py

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

app = modal.App("a2rag-perquery-targets", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("thenlper/gte-small", "general"),
]

DATASETS = ["scifact", "arguana"]

GENERIC_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}

ALPHA_MAX = 0.2
RRF_K = 60
MAX_QUERIES = 100  # More queries for statistical power


@app.function(gpu="L4", memory=32768, timeout=5400, volumes={"/results": vol})
def run_perquery_targets(model_name: str, family: str):
    """Full adaptive stack with LLM-generated per-query targets."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc

    print(f"\n{'='*60}")
    print(f"  Per-Query Targets + Full Stack: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    # Load LLM for target generation
    print("  Loading Qwen2.5-3B-Instruct for target generation...")
    llm_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float16, device_map="auto"
    )

    def generate_targets(query: str, domain: str, n_targets: int = 3) -> list:
        """Generate cross-domain exploration targets for a query."""
        prompt = f"""You are a research librarian. Given a query from {domain}, suggest {n_targets} SHORT phrases (3-6 words each) describing adjacent domains or perspectives that could provide relevant insights.

Query: {query[:300]}

Return ONLY {n_targets} phrases, one per line, no numbering or explanation:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm.generate(
                **inputs, max_new_tokens=80, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Parse targets (one per line)
        targets = [line.strip().strip("-•·").strip() for line in response.split("\n") if line.strip() and len(line.strip()) > 3]
        return targets[:n_targets]

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def normalize_vec(v):
        n = np.linalg.norm(v)
        return v / max(n, 1e-10)

    def remove_top_k(embs, k=1):
        """Remove top-k principal components."""
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(embs)
        result = embs.copy()
        for comp in pca.components_[:k]:
            result -= (result @ comp).reshape(-1, 1) * comp
        return normalize_rows(result), pca.components_[:k]

    def rrf_fusion(scores_list, k=60):
        """RRF fusion of multiple score arrays."""
        n_docs = scores_list[0].shape[0]
        rrf_scores = np.zeros(n_docs)
        for scores in scores_list:
            ranks = np.argsort(np.argsort(-scores)) + 1
            rrf_scores += 1.0 / (k + ranks)
        return rrf_scores

    model_results = {"model": model_name, "family": family}

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")

        domain_map = {"scifact": "biomedical research", "arguana": "argumentation and debate"}
        domain = domain_map[ds_name]

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())[:MAX_QUERIES]
        query_texts = [queries[q] for q in query_ids]

        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        # Encode corpus and queries
        corpus_embs_raw = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs_raw = np.array(embed_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))

        # Top-k removal for isotropy correction
        corpus_embs_tk, removed_components = remove_top_k(corpus_embs_raw, k=1)
        query_embs_tk = query_embs_raw.copy()
        for comp in removed_components:
            query_embs_tk -= (query_embs_tk @ comp).reshape(-1, 1) * comp
        query_embs_tk = normalize_rows(query_embs_tk)

        # Generic target embedding
        generic_target = embed_model.encode(GENERIC_TARGETS[ds_name], normalize_embeddings=True)
        generic_target_tk = generic_target.copy()
        for comp in removed_components:
            generic_target_tk -= np.dot(generic_target_tk, comp) * comp
        generic_target_tk = normalize_vec(generic_target_tk)

        def eval_ndcg(retrieval_results):
            ndcg, _, _, _ = evaluator.evaluate(qrels, retrieval_results, [1, 5, 10])
            return ndcg

        def get_results_from_scores(scores_matrix):
            results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(scores_matrix[i])[::-1][:100]
                results[qid] = {doc_ids[idx]: float(scores_matrix[i, idx]) for idx in top}
            return results

        # === A. Baseline (original embeddings) ===
        sims_base = query_embs_raw @ corpus_embs_raw.T
        ndcg_base = eval_ndcg(get_results_from_scores(sims_base))
        base_10 = ndcg_base.get("NDCG@10", 0)
        print(f"    A. Baseline: nDCG@10={base_10:.4f}")

        # === B. Addition uniforme (generic target, fixed alpha) ===
        q_add = query_embs_raw + ALPHA_MAX * generic_target
        q_add = normalize_rows(q_add)
        sims_add = q_add @ corpus_embs_raw.T
        ndcg_add = eval_ndcg(get_results_from_scores(sims_add))
        delta_add = ndcg_add.get("NDCG@10", 0) - base_10
        print(f"    B. Addition uniforme: nDCG@10={ndcg_add.get('NDCG@10',0):.4f} (Δ={delta_add:+.4f})")

        # === C. Multi-vector generic target ===
        sims_base_raw = query_embs_raw @ corpus_embs_raw.T
        sims_add_raw = q_add @ corpus_embs_raw.T
        mv_generic_results = {}
        for i, qid in enumerate(query_ids):
            rrf = rrf_fusion([sims_base_raw[i], sims_add_raw[i]], k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            mv_generic_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}
        ndcg_mv_generic = eval_ndcg(mv_generic_results)
        delta_mv_generic = ndcg_mv_generic.get("NDCG@10", 0) - base_10
        print(f"    C. Multi-vector generic: nDCG@10={ndcg_mv_generic.get('NDCG@10',0):.4f} (Δ={delta_mv_generic:+.4f})")

        # === D. Multi-vector with PER-QUERY targets ===
        print(f"    Generating per-query targets...")
        all_perquery_targets = {}
        for idx, (qid, q_text) in enumerate(zip(query_ids, query_texts)):
            targets = generate_targets(q_text, domain, n_targets=3)
            all_perquery_targets[qid] = targets
            if idx % 20 == 0:
                print(f"      Generated targets for {idx+1}/{len(query_ids)} queries")

        # Encode all unique targets
        unique_targets = set()
        for targets in all_perquery_targets.values():
            unique_targets.update(targets)
        unique_targets = list(unique_targets)
        print(f"    Unique targets: {len(unique_targets)}")

        target_embs = {}
        if unique_targets:
            embs = np.array(embed_model.encode(unique_targets, normalize_embeddings=True, show_progress_bar=False))
            for t, e in zip(unique_targets, embs):
                target_embs[t] = e

        # Multi-vector with per-query targets (on RAW embeddings)
        mv_perquery_results = {}
        new_docs_per_query = []
        for i, qid in enumerate(query_ids):
            targets = all_perquery_targets[qid]
            score_lists = [sims_base_raw[i]]

            for t_text in targets:
                if t_text not in target_embs:
                    continue
                t_emb = target_embs[t_text]
                sim_qt = float(query_embs_raw[i] @ t_emb)
                alpha = ALPHA_MAX * (1 - sim_qt) ** 2  # Adaptive alpha
                q_rot = normalize_vec(query_embs_raw[i] + alpha * t_emb)
                score_lists.append(q_rot @ corpus_embs_raw.T)

            rrf = rrf_fusion(score_lists, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            mv_perquery_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}

            # Count new docs
            base_top = set(doc_ids[idx] for idx in np.argsort(sims_base_raw[i])[::-1][:10])
            new_top = set(doc_ids[idx] for idx in top[:10])
            new_docs_per_query.append(len(new_top - base_top))

        ndcg_mv_pq = eval_ndcg(mv_perquery_results)
        delta_mv_pq = ndcg_mv_pq.get("NDCG@10", 0) - base_10
        avg_new_docs = np.mean(new_docs_per_query)
        print(f"    D. Multi-vector per-query: nDCG@10={ndcg_mv_pq.get('NDCG@10',0):.4f} (Δ={delta_mv_pq:+.4f})"
              f"  new_docs_avg={avg_new_docs:.1f}")

        # === E. Full stack: top-k + adaptive alpha + multi-vector per-query ===
        # Use isotropy-corrected embeddings
        sims_base_tk = query_embs_tk @ corpus_embs_tk.T
        mv_stack_results = {}
        for i, qid in enumerate(query_ids):
            targets = all_perquery_targets[qid]
            score_lists = [sims_base_tk[i]]

            for t_text in targets:
                if t_text not in target_embs:
                    continue
                t_emb = target_embs[t_text].copy()
                # Apply top-k removal to target too
                for comp in removed_components:
                    t_emb -= np.dot(t_emb, comp) * comp
                t_emb = normalize_vec(t_emb)

                sim_qt = float(query_embs_tk[i] @ t_emb)
                alpha = ALPHA_MAX * (1 - sim_qt) ** 2
                q_rot = normalize_vec(query_embs_tk[i] + alpha * t_emb)
                score_lists.append(q_rot @ corpus_embs_tk.T)

            rrf = rrf_fusion(score_lists, k=RRF_K)
            top = np.argsort(rrf)[::-1][:100]
            mv_stack_results[qid] = {doc_ids[idx]: float(rrf[idx]) for idx in top}

        ndcg_stack = eval_ndcg(mv_stack_results)
        delta_stack = ndcg_stack.get("NDCG@10", 0) - base_10
        print(f"    E. Full stack (top-k+adapt+mv_pq): nDCG@10={ndcg_stack.get('NDCG@10',0):.4f} (Δ={delta_stack:+.4f})")

        # Save per-query targets sample
        target_sample = {qid: all_perquery_targets[qid] for qid in query_ids[:10]}

        ds_results = {
            "n_queries": len(query_ids),
            "n_docs": len(doc_ids),
            "A_baseline": {k: round(v, 4) for k, v in ndcg_base.items()},
            "B_addition_uniform": {
                "ndcg": {k: round(v, 4) for k, v in ndcg_add.items()},
                "delta": round(delta_add, 4),
            },
            "C_multivector_generic": {
                "ndcg": {k: round(v, 4) for k, v in ndcg_mv_generic.items()},
                "delta": round(delta_mv_generic, 4),
            },
            "D_multivector_perquery": {
                "ndcg": {k: round(v, 4) for k, v in ndcg_mv_pq.items()},
                "delta": round(delta_mv_pq, 4),
                "new_docs_avg_top10": round(avg_new_docs, 2),
                "unique_targets": len(unique_targets),
            },
            "E_full_stack": {
                "ndcg": {k: round(v, 4) for k, v in ndcg_stack.items()},
                "delta": round(delta_stack, 4),
            },
            "target_samples": target_sample,
        }

        model_results[ds_name] = ds_results

    # Save
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/perquery_targets/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Per-Query Targets + Full Adaptive Stack")
    print("  3 models x 2 datasets x 5 strategies")
    print("=" * 70)

    all_results = list(run_perquery_targets.starmap(
        [(name, fam) for name, fam in MODELS]
    ))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: nDCG@10 deltas vs baseline")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Base':>8} {'B.Unif':>8} {'C.MV_gen':>8} {'D.MV_pq':>8} {'E.Stack':>8}")
    print("  " + "-" * 82)

    for r in all_results:
        for ds in DATASETS:
            if ds not in r:
                continue
            d = r[ds]
            base = d["A_baseline"].get("NDCG@10", 0)
            print(f"  {r['model']:<30} {ds:<10} {base:>8.4f}"
                  f" {d['B_addition_uniform']['delta']:>+8.4f}"
                  f" {d['C_multivector_generic']['delta']:>+8.4f}"
                  f" {d['D_multivector_perquery']['delta']:>+8.4f}"
                  f" {d['E_full_stack']['delta']:>+8.4f}")

    print("\n  DONE!")
