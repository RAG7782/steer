"""
A2RAG — Pergunta 2: Rotacionar o corpus em vez da query?

Hipotese: Em vez de T(q), criar D' = {T(d) for d in D} — um indice "rotacionado".
Buscar q em D (original) e D' (rotacionado). Merge dos dois top-K.
Se funcionar, elimina o trade-off completamente (zero custo nDCG no indice original).

Setup:
1. Indexar corpus original D
2. Criar corpus rotacionado D' = {normalize(d + alpha * target) for d in D}
3. Buscar q em D -> top-K1
4. Buscar q em D' -> top-K2
5. Merge + deduplicate -> top-K final
6. Medir nDCG@10

Usage: modal run modal_rotated_corpus.py

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

app = modal.App("a2rag-rotated-corpus", image=image)
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

ALPHAS = [0.05, 0.1, 0.2, 0.3]


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_rotated_corpus(model_name: str, params_m: int, family: str):
    """Run dual-index experiment for ONE model on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Rotated Corpus: {model_name} ({params_m}M, {family})")
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

        # Encode everything
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        # Helper: evaluate nDCG@10 given query_embs and corpus_embs
        def eval_ndcg(q_embs, c_embs, top_k=100):
            sims = q_embs @ c_embs.T
            results = {}
            for i, qid in enumerate(query_ids):
                top_indices = np.argsort(sims[i])[::-1][:top_k]
                results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top_indices}
            ndcg, _, _, _ = evaluator.evaluate(qrels, results, [1, 5, 10])
            return ndcg

        # Baseline: q -> D
        baseline_ndcg = eval_ndcg(query_embs, corpus_embs)
        print(f"    Baseline nDCG@10: {baseline_ndcg.get('NDCG@10', 0):.4f}")

        # For each alpha, create rotated corpus D' and do dual-index
        ds_results = {"baseline": baseline_ndcg}

        for alpha in ALPHAS:
            # Rotate corpus: D' = {normalize(d + alpha * target) for d in D}
            corpus_rotated = corpus_embs + alpha * target_emb
            norms = np.linalg.norm(corpus_rotated, axis=1, keepdims=True)
            corpus_rotated = corpus_rotated / np.maximum(norms, 1e-10)

            # Also test: rotate query (original approach) for comparison
            query_rotated = query_embs + alpha * target_emb
            q_norms = np.linalg.norm(query_rotated, axis=1, keepdims=True)
            query_rotated = query_rotated / np.maximum(q_norms, 1e-10)

            # --- Strategy A: Query rotation (original) ---
            ndcg_query_rot = eval_ndcg(query_rotated, corpus_embs)

            # --- Strategy B: Dual-index (q -> D original + q -> D') ---
            # Search original index
            sims_orig = query_embs @ corpus_embs.T
            # Search rotated index
            sims_rot = query_embs @ corpus_rotated.T

            # Merge: for each query, take union of top-100 from each, keep max score
            merged_results = {}
            for i, qid in enumerate(query_ids):
                top_orig = np.argsort(sims_orig[i])[::-1][:100]
                top_rot = np.argsort(sims_rot[i])[::-1][:100]

                # Merge with max score (original index scores used for ranking)
                doc_scores = {}
                for idx in top_orig:
                    did = doc_ids[idx]
                    doc_scores[did] = float(sims_orig[i, idx])
                for idx in top_rot:
                    did = doc_ids[idx]
                    # Use original similarity for fair comparison with baseline
                    orig_sim = float(sims_orig[i, idx])
                    if did not in doc_scores or orig_sim > doc_scores[did]:
                        doc_scores[did] = orig_sim

                merged_results[qid] = doc_scores

            ndcg_dual = evaluator.evaluate(qrels, merged_results, [1, 5, 10])[0]

            # --- Strategy C: Dual-index with rotated-index scores for new docs ---
            merged_results_c = {}
            for i, qid in enumerate(query_ids):
                top_orig = set(np.argsort(sims_orig[i])[::-1][:100].tolist())
                top_rot = np.argsort(sims_rot[i])[::-1][:100]

                doc_scores = {}
                for idx in np.argsort(sims_orig[i])[::-1][:100]:
                    doc_scores[doc_ids[idx]] = float(sims_orig[i, idx])
                for idx in top_rot:
                    did = doc_ids[idx]
                    if did not in doc_scores:
                        # New doc surfaced by rotated index — use rotated score
                        doc_scores[did] = float(sims_rot[i, idx])

                merged_results_c[qid] = doc_scores

            ndcg_dual_c = evaluator.evaluate(qrels, merged_results_c, [1, 5, 10])[0]

            # Count how many NEW docs the rotated index surfaces
            new_docs_counts = []
            for i, qid in enumerate(query_ids):
                orig_set = set(doc_ids[idx] for idx in np.argsort(sims_orig[i])[::-1][:100])
                rot_set = set(doc_ids[idx] for idx in np.argsort(sims_rot[i])[::-1][:100])
                new_docs_counts.append(len(rot_set - orig_set))

            avg_new_docs = np.mean(new_docs_counts)
            max_new_docs = max(new_docs_counts)

            delta_query_rot = ndcg_query_rot.get("NDCG@10", 0) - baseline_ndcg.get("NDCG@10", 0)
            delta_dual = ndcg_dual.get("NDCG@10", 0) - baseline_ndcg.get("NDCG@10", 0)
            delta_dual_c = ndcg_dual_c.get("NDCG@10", 0) - baseline_ndcg.get("NDCG@10", 0)

            print(f"    alpha={alpha}: query_rot={ndcg_query_rot.get('NDCG@10',0):.4f} (Δ={delta_query_rot:+.4f})"
                  f"  dual_orig_scores={ndcg_dual.get('NDCG@10',0):.4f} (Δ={delta_dual:+.4f})"
                  f"  dual_mixed_scores={ndcg_dual_c.get('NDCG@10',0):.4f} (Δ={delta_dual_c:+.4f})"
                  f"  new_docs_avg={avg_new_docs:.1f} max={max_new_docs}")

            ds_results[f"alpha_{alpha}"] = {
                "query_rotation": {k: round(v, 4) for k, v in ndcg_query_rot.items()},
                "dual_index_orig_scores": {k: round(v, 4) for k, v in ndcg_dual.items()},
                "dual_index_mixed_scores": {k: round(v, 4) for k, v in ndcg_dual_c.items()},
                "delta_query_rot": round(delta_query_rot, 4),
                "delta_dual_orig": round(delta_dual, 4),
                "delta_dual_mixed": round(delta_dual_c, 4),
                "new_docs_avg": round(avg_new_docs, 1),
                "new_docs_max": max_new_docs,
            }

        model_results[ds_name] = ds_results

    # Save results
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/rotated_corpus/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Pergunta 2: Rotated Corpus (Dual-Index)")
    print("  6 models x 2 datasets x 4 alphas x 3 strategies")
    print("=" * 70)

    all_results = list(run_rotated_corpus.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: nDCG@10 deltas vs baseline (alpha=0.1)")
    print("=" * 70)
    print(f"  {'Model':<30} {'Dataset':<10} {'QueryRot':>10} {'DualOrig':>10} {'DualMixed':>10} {'NewDocs':>8}")
    print("  " + "-" * 78)

    for r in all_results:
        for ds in DATASETS:
            if ds in r and "alpha_0.1" in r[ds]:
                a = r[ds]["alpha_0.1"]
                print(f"  {r['model']:<30} {ds:<10} {a['delta_query_rot']:>+10.4f} {a['delta_dual_orig']:>+10.4f} {a['delta_dual_mixed']:>+10.4f} {a['new_docs_avg']:>8.1f}")

    print("\n  DONE!")
