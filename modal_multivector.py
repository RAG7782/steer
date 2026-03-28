"""
A2RAG — Multi-Vector Query: q + T(q)

Hipotese: Em vez de substituir q por T(q), buscar AMBOS contra o mesmo indice D.
q preserva o baseline. T(q) surfaca docs novos. Fusion dos scores garante
que nenhum doc relevante do baseline e perdido.

Estrategias de fusion:
1. Max-score: score(d) = max(sim(q,d), lambda * sim(T(q),d))
2. Sum-score: score(d) = sim(q,d) + lambda * sim(T(q),d)
3. RRF (Reciprocal Rank Fusion): 1/(k+rank_q) + 1/(k+rank_Tq)
4. Cascade: top-100 de q, union top-100 de T(q), re-rank por sim(q,d)

Usage: modal run modal_multivector.py

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
    )
)

app = modal.App("a2rag-multivector", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", 22, "distilled"),
    ("BAAI/bge-small-en-v1.5", 33, "contrastive"),
    ("all-mpnet-base-v2", 109, "trained-1B-pairs"),
    ("BAAI/bge-base-en-v1.5", 109, "contrastive"),
    ("intfloat/e5-small-v2", 33, "instruction-tuned"),
    ("thenlper/gte-small", 33, "general-text"),
]

DATASETS = ["scifact", "arguana"]

ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}

ALPHAS = [0.1, 0.2, 0.3]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
RRF_K = 60


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_multivector(model_name: str, params_m: int, family: str):
    """Test multi-vector query strategies for ONE model on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Multi-Vector: {model_name} ({params_m}M, {family})")
    print(f"{'='*60}")

    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    model_results = {"model": model_name, "params_m": params_m, "family": family}

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        def eval_results(results_dict):
            ndcg, _, _, _ = evaluator.evaluate(qrels, results_dict, [1, 5, 10])
            return ndcg

        # Baseline similarities
        sims_q = query_embs @ corpus_embs.T  # (n_queries, n_docs)

        # Baseline nDCG
        baseline_results = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims_q[i])[::-1][:100]
            baseline_results[qid] = {doc_ids[idx]: float(sims_q[i, idx]) for idx in top}
        ndcg_base = eval_results(baseline_results)
        base_10 = ndcg_base.get("NDCG@10", 0)
        print(f"    Baseline nDCG@10: {base_10:.4f}")

        ds_results = {"baseline": {k: round(v, 4) for k, v in ndcg_base.items()}}

        for alpha in ALPHAS:
            # Compute T(q) = normalize(q + alpha * target)
            q_rot = query_embs + alpha * target_emb
            q_rot_norms = np.linalg.norm(q_rot, axis=1, keepdims=True)
            q_rot = q_rot / np.maximum(q_rot_norms, 1e-10)

            sims_tq = q_rot @ corpus_embs.T  # (n_queries, n_docs)

            # === Addition-only (reference) ===
            add_results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims_tq[i])[::-1][:100]
                add_results[qid] = {doc_ids[idx]: float(sims_tq[i, idx]) for idx in top}
            ndcg_add = eval_results(add_results)
            delta_add = ndcg_add.get("NDCG@10", 0) - base_10

            alpha_results = {
                "addition_only": {"ndcg": {k: round(v, 4) for k, v in ndcg_add.items()}, "delta": round(delta_add, 4)},
            }

            best_delta = delta_add
            best_strat = f"addition_only"

            for lam in LAMBDAS:
                # === Max-score fusion ===
                max_results = {}
                for i, qid in enumerate(query_ids):
                    scores = np.maximum(sims_q[i], lam * sims_tq[i])
                    top = np.argsort(scores)[::-1][:100]
                    max_results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top}
                ndcg_max = eval_results(max_results)
                delta_max = ndcg_max.get("NDCG@10", 0) - base_10

                # === Sum-score fusion ===
                sum_results = {}
                for i, qid in enumerate(query_ids):
                    scores = sims_q[i] + lam * sims_tq[i]
                    top = np.argsort(scores)[::-1][:100]
                    sum_results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top}
                ndcg_sum = eval_results(sum_results)
                delta_sum = ndcg_sum.get("NDCG@10", 0) - base_10

                # === RRF ===
                rrf_results = {}
                for i, qid in enumerate(query_ids):
                    ranks_q = np.argsort(np.argsort(-sims_q[i])) + 1  # rank from 1
                    ranks_tq = np.argsort(np.argsort(-sims_tq[i])) + 1
                    rrf_scores = 1.0 / (RRF_K + ranks_q) + lam * 1.0 / (RRF_K + ranks_tq)
                    top = np.argsort(rrf_scores)[::-1][:100]
                    rrf_results[qid] = {doc_ids[idx]: float(rrf_scores[idx]) for idx in top}
                ndcg_rrf = eval_results(rrf_results)
                delta_rrf = ndcg_rrf.get("NDCG@10", 0) - base_10

                print(f"    α={alpha} λ={lam}: add={delta_add:+.4f}  max={delta_max:+.4f}  sum={delta_sum:+.4f}  rrf={delta_rrf:+.4f}")

                alpha_results[f"max_l{lam}"] = {"ndcg": {k: round(v, 4) for k, v in ndcg_max.items()}, "delta": round(delta_max, 4)}
                alpha_results[f"sum_l{lam}"] = {"ndcg": {k: round(v, 4) for k, v in ndcg_sum.items()}, "delta": round(delta_sum, 4)}
                alpha_results[f"rrf_l{lam}"] = {"ndcg": {k: round(v, 4) for k, v in ndcg_rrf.items()}, "delta": round(delta_rrf, 4)}

                for d, s in [(delta_max, f"max_l{lam}"), (delta_sum, f"sum_l{lam}"), (delta_rrf, f"rrf_l{lam}")]:
                    if d > best_delta:
                        best_delta = d
                        best_strat = s

            # === Cascade: top-100(q) ∪ top-100(T(q)), re-rank by sim(q) ===
            cascade_results = {}
            new_docs_counts = []
            for i, qid in enumerate(query_ids):
                top_q = set(np.argsort(sims_q[i])[::-1][:100].tolist())
                top_tq = set(np.argsort(sims_tq[i])[::-1][:100].tolist())
                union = top_q | top_tq
                new_docs_counts.append(len(top_tq - top_q))
                # Re-rank by original q similarity
                doc_scores = {doc_ids[idx]: float(sims_q[i, idx]) for idx in union}
                # Keep top 100
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
                cascade_results[qid] = dict(sorted_docs)
            ndcg_cascade = eval_results(cascade_results)
            delta_cascade = ndcg_cascade.get("NDCG@10", 0) - base_10

            print(f"    α={alpha} cascade: Δ={delta_cascade:+.4f}  new_docs_avg={np.mean(new_docs_counts):.1f}")

            alpha_results["cascade"] = {
                "ndcg": {k: round(v, 4) for k, v in ndcg_cascade.items()},
                "delta": round(delta_cascade, 4),
                "new_docs_avg": round(float(np.mean(new_docs_counts)), 1),
            }
            if delta_cascade > best_delta:
                best_delta = delta_cascade
                best_strat = "cascade"

            alpha_results["best"] = {"strategy": best_strat, "delta": round(best_delta, 4)}
            ds_results[f"alpha_{alpha}"] = alpha_results

        model_results[ds_name] = ds_results

    safe_name = model_name.replace("/", "_")
    out_path = f"/results/multivector/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Multi-Vector Query: q ⊕ T(q)")
    print("  6 models x 2 datasets x 3 alphas x (4 lambdas x 3 fusions + cascade)")
    print("=" * 70)

    all_results = list(run_multivector.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Best multi-vector strategy vs addition-only (alpha=0.1)")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Add':>8} {'Best_MV':>8} {'Strategy':<16}")
    print("  " + "-" * 72)

    for r in all_results:
        for ds in DATASETS:
            if ds not in r or "alpha_0.1" not in r[ds]:
                continue
            a = r[ds]["alpha_0.1"]
            add_d = a["addition_only"]["delta"]
            best = a.get("best", {})
            print(f"  {r['model']:<30} {ds:<10} {add_d:>+8.4f} {best.get('delta',0):>+8.4f} {best.get('strategy','?'):<16}")

    print("\n  DONE!")
