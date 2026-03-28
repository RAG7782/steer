"""
A2RAG — Pergunta 1: Addition com Reranker elimina o trade-off?

Hipotese: Addition amplia recall (traz docs novos do dominio-alvo),
mas pode degradar nDCG porque docs irrelevantes entram no top-K.
Um cross-encoder reranker pode filtrar o ruido e preservar o ganho.

Pipeline:
1. Busca padrao q -> top-100 (bi-encoder)
2. Busca rotacionada T(q) -> top-100 (bi-encoder)
3. Merge union -> ~120-150 docs unicos
4. Rerank com cross-encoder -> top-10 final
5. Medir nDCG@10

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (rápido, SOTA para rerank)

Usage: modal run modal_reranker_test.py

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

app = modal.App("a2rag-reranker", image=image)
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
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_reranker_test(model_name: str, params_m: int, family: str):
    """Test Addition + Reranker for ONE bi-encoder on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Reranker Test: {model_name} ({params_m}M, {family})")
    print(f"{'='*60}")

    bi_encoder = SentenceTransformer(model_name)
    cross_encoder = CrossEncoder(CROSS_ENCODER)
    evaluator = EvaluateRetrieval()
    model_results = {"model": model_name, "params_m": params_m, "family": family, "cross_encoder": CROSS_ENCODER}

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        # Build doc_id -> text map for reranking
        id_to_text = {doc_ids[i]: doc_texts[i] for i in range(len(doc_ids))}

        # Encode
        corpus_embs = np.array(bi_encoder.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(bi_encoder.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = bi_encoder.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        # Baseline cosine similarity
        sims_baseline = query_embs @ corpus_embs.T

        def get_top_k(sims, k=100):
            """Return dict: qid -> {doc_id: score}"""
            results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:k]
                results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            return results

        def rerank(candidate_results, top_k=10):
            """Rerank candidates with cross-encoder, return top_k."""
            reranked = {}
            # Batch all pairs
            all_pairs = []
            all_meta = []  # (qid, doc_id)
            for qid in query_ids:
                q_text = queries[qid]
                for did in candidate_results.get(qid, {}):
                    all_pairs.append((q_text, id_to_text[did]))
                    all_meta.append((qid, did))

            if not all_pairs:
                return {}

            # Score in batches
            scores = cross_encoder.predict(all_pairs, batch_size=256, show_progress_bar=False)

            # Group by query
            query_scores = {}
            for (qid, did), score in zip(all_meta, scores):
                query_scores.setdefault(qid, {})[did] = float(score)

            # Take top-k per query
            for qid, doc_scores in query_scores.items():
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                reranked[qid] = dict(sorted_docs)

            return reranked

        # === 1. Baseline (bi-encoder only) ===
        baseline_results = get_top_k(sims_baseline, 100)
        ndcg_baseline = evaluator.evaluate(qrels, baseline_results, [1, 5, 10])[0]
        print(f"    Baseline nDCG@10: {ndcg_baseline.get('NDCG@10', 0):.4f}")

        # === 2. Baseline + Reranker ===
        reranked_baseline = rerank(baseline_results, top_k=100)
        ndcg_baseline_reranked = evaluator.evaluate(qrels, reranked_baseline, [1, 5, 10])[0]
        print(f"    Baseline+Reranker nDCG@10: {ndcg_baseline_reranked.get('NDCG@10', 0):.4f}")

        ds_results = {
            "baseline": {k: round(v, 4) for k, v in ndcg_baseline.items()},
            "baseline_reranked": {k: round(v, 4) for k, v in ndcg_baseline_reranked.items()},
        }

        for alpha in ALPHAS:
            # Addition: T(q) = normalize(q + alpha * target)
            query_add = query_embs + alpha * target_emb
            q_norms = np.linalg.norm(query_add, axis=1, keepdims=True)
            query_add = query_add / np.maximum(q_norms, 1e-10)
            sims_add = query_add @ corpus_embs.T

            # === 3. Addition only (no reranker) ===
            add_results = get_top_k(sims_add, 100)
            ndcg_add = evaluator.evaluate(qrels, add_results, [1, 5, 10])[0]

            # === 4. Addition + Reranker ===
            reranked_add = rerank(add_results, top_k=100)
            ndcg_add_reranked = evaluator.evaluate(qrels, reranked_add, [1, 5, 10])[0]

            # === 5. Merged (baseline union addition) + Reranker ===
            merged = {}
            for qid in query_ids:
                docs = dict(baseline_results.get(qid, {}))
                for did, score in add_results.get(qid, {}).items():
                    if did not in docs or score > docs[did]:
                        docs[did] = score
                merged[qid] = docs

            reranked_merged = rerank(merged, top_k=100)
            ndcg_merged_reranked = evaluator.evaluate(qrels, reranked_merged, [1, 5, 10])[0]

            # Count merged candidates
            merged_sizes = [len(merged[qid]) for qid in query_ids]
            avg_candidates = np.mean(merged_sizes)

            delta_add = ndcg_add.get("NDCG@10", 0) - ndcg_baseline.get("NDCG@10", 0)
            delta_add_reranked = ndcg_add_reranked.get("NDCG@10", 0) - ndcg_baseline.get("NDCG@10", 0)
            delta_merged_reranked = ndcg_merged_reranked.get("NDCG@10", 0) - ndcg_baseline.get("NDCG@10", 0)

            print(f"    alpha={alpha}: add={ndcg_add.get('NDCG@10',0):.4f}(Δ{delta_add:+.4f})"
                  f"  add+rerank={ndcg_add_reranked.get('NDCG@10',0):.4f}(Δ{delta_add_reranked:+.4f})"
                  f"  merged+rerank={ndcg_merged_reranked.get('NDCG@10',0):.4f}(Δ{delta_merged_reranked:+.4f})"
                  f"  candidates={avg_candidates:.0f}")

            ds_results[f"alpha_{alpha}"] = {
                "addition_only": {k: round(v, 4) for k, v in ndcg_add.items()},
                "addition_reranked": {k: round(v, 4) for k, v in ndcg_add_reranked.items()},
                "merged_reranked": {k: round(v, 4) for k, v in ndcg_merged_reranked.items()},
                "delta_addition": round(delta_add, 4),
                "delta_addition_reranked": round(delta_add_reranked, 4),
                "delta_merged_reranked": round(delta_merged_reranked, 4),
                "avg_candidates_merged": round(avg_candidates, 1),
            }

        model_results[ds_name] = ds_results

    safe_name = model_name.replace("/", "_")
    out_path = f"/results/reranker_test/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Pergunta 1: Addition + Reranker")
    print("  6 models x 2 datasets x 3 alphas x 5 strategies")
    print("=" * 70)

    all_results = list(run_reranker_test.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: nDCG@10 (alpha=0.1)")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Base':>8} {'Base+RR':>8} {'Add':>8} {'Add+RR':>8} {'Mrg+RR':>8}")
    print("  " + "-" * 82)

    for r in all_results:
        for ds in DATASETS:
            if ds in r:
                b = r[ds]["baseline"].get("NDCG@10", 0)
                br = r[ds]["baseline_reranked"].get("NDCG@10", 0)
                a = r[ds].get("alpha_0.1", {})
                ao = a.get("addition_only", {}).get("NDCG@10", 0)
                ar = a.get("addition_reranked", {}).get("NDCG@10", 0)
                mr = a.get("merged_reranked", {}).get("NDCG@10", 0)
                print(f"  {r['model']:<30} {ds:<10} {b:>8.4f} {br:>8.4f} {ao:>8.4f} {ar:>8.4f} {mr:>8.4f}")

    print("\n  DONE!")
