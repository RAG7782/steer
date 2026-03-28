"""
A2RAG — Pergunta 3: Addition + Scaled Subtraction composta

Hipotese: Combinar Addition (puxa para dominio-alvo) com Subtraction escalada
(repele do dominio-fonte), criando uma operacao composta mais potente que cada uma isolada.

Operacao:
  T(q) = normalize(q + alpha * target - beta * proj_exclude(q))

onde proj_exclude(q) = (q . exclude) * exclude  (projecao ortogonal)

Isto combina o "puxar" (addition) com o "empurrar" (subtraction) num unico passo.

Hipotese-chave: a subtraction nao degradou nDCG tanto (Table 5 do paper).
Entao podemos usar um beta agressivo (0.25-0.5) para "limpar" a query
antes de adicionar o target.

Grid: alpha={0.05, 0.1, 0.2} x beta={0.0, 0.25, 0.5, 0.75} x 6 modelos x 2 datasets

Usage: modal run modal_composed_addition.py

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

app = modal.App("a2rag-composed-addition", image=image)
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

EXCLUSION_CONCEPTS = {
    "scifact": "methodology and statistical analysis",
    "arguana": "informal argumentation and rhetoric",
}

ALPHAS = [0.05, 0.1, 0.2]
BETAS = [0.0, 0.25, 0.5, 0.75]


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_composed(model_name: str, params_m: int, family: str):
    """Test composed Addition+ScaledSubtraction for ONE model on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Composed Add+Sub: {model_name} ({params_m}M, {family})")
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

        # Encode
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)
        exclude_emb = model.encode(EXCLUSION_CONCEPTS[ds_name], normalize_embeddings=True)

        def eval_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, results, [1, 5, 10])
            return ndcg

        # Baseline
        ndcg_base = eval_ndcg(query_embs, corpus_embs)
        base_10 = ndcg_base.get("NDCG@10", 0)
        print(f"    Baseline nDCG@10: {base_10:.4f}")

        ds_results = {"baseline": {k: round(v, 4) for k, v in ndcg_base.items()}}

        best_delta = -999
        best_config = ""

        for alpha in ALPHAS:
            for beta in BETAS:
                # Composed operation: T(q) = normalize(q + alpha*target - beta*proj(q, exclude))
                # proj(q, exclude) = (q . exclude) * exclude
                proj = (query_embs @ exclude_emb).reshape(-1, 1) * exclude_emb
                transformed = query_embs + alpha * target_emb - beta * proj

                # Normalize
                norms = np.linalg.norm(transformed, axis=1, keepdims=True)
                transformed = transformed / np.maximum(norms, 1e-10)

                ndcg = eval_ndcg(transformed, corpus_embs)
                delta = ndcg.get("NDCG@10", 0) - base_10

                # Also compute individual ops for comparison
                # Addition only
                if beta == 0.0:
                    label = f"add_only_a{alpha}"
                else:
                    label = f"a{alpha}_b{beta}"

                # Track similarity shifts
                mean_sim_target = float(np.mean(transformed @ target_emb))
                mean_sim_exclude = float(np.mean(np.abs(transformed @ exclude_emb)))
                orig_sim_target = float(np.mean(query_embs @ target_emb))
                orig_sim_exclude = float(np.mean(np.abs(query_embs @ exclude_emb)))

                print(f"    α={alpha} β={beta}: nDCG@10={ndcg.get('NDCG@10',0):.4f} (Δ={delta:+.4f})"
                      f"  sim_target: {orig_sim_target:.3f}→{mean_sim_target:.3f}"
                      f"  sim_exclude: {orig_sim_exclude:.3f}→{mean_sim_exclude:.3f}")

                ds_results[f"a{alpha}_b{beta}"] = {
                    "ndcg": {k: round(v, 4) for k, v in ndcg.items()},
                    "delta_10": round(delta, 4),
                    "sim_target_before": round(orig_sim_target, 4),
                    "sim_target_after": round(mean_sim_target, 4),
                    "sim_exclude_before": round(orig_sim_exclude, 4),
                    "sim_exclude_after": round(mean_sim_exclude, 4),
                }

                if delta > best_delta:
                    best_delta = delta
                    best_config = f"α={alpha} β={beta}"

        # Also test subtraction-only for reference
        proj = (query_embs @ exclude_emb).reshape(-1, 1) * exclude_emb
        sub_only = query_embs - proj
        sub_norms = np.linalg.norm(sub_only, axis=1, keepdims=True)
        sub_only = sub_only / np.maximum(sub_norms, 1e-10)
        ndcg_sub = eval_ndcg(sub_only, corpus_embs)
        ds_results["subtraction_only"] = {k: round(v, 4) for k, v in ndcg_sub.items()}

        print(f"    Subtraction-only nDCG@10: {ndcg_sub.get('NDCG@10',0):.4f} (Δ={ndcg_sub.get('NDCG@10',0)-base_10:+.4f})")
        print(f"    >>> BEST: {best_config} Δ={best_delta:+.4f}")

        ds_results["best"] = {"config": best_config, "delta": round(best_delta, 4)}
        model_results[ds_name] = ds_results

    safe_name = model_name.replace("/", "_")
    out_path = f"/results/composed_addition/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Pergunta 3: Composed Addition + Scaled Subtraction")
    print("  6 models x 2 datasets x 3 alphas x 4 betas = 144 configs")
    print("=" * 70)

    all_results = list(run_composed.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Summary: best config per model per dataset
    print("\n" + "=" * 70)
    print("  SUMMARY: Best composed config vs baseline & add-only (alpha=0.1)")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Base':>8} {'Add0.1':>8} {'BestComp':>8} {'Config':<16}")
    print("  " + "-" * 82)

    for r in all_results:
        for ds in DATASETS:
            if ds in r:
                base = r[ds]["baseline"].get("NDCG@10", 0)
                add01 = r[ds].get("a0.1_b0.0", {}).get("ndcg", {}).get("NDCG@10", 0)
                best = r[ds].get("best", {})
                best_ndcg = base + best.get("delta", 0)
                print(f"  {r['model']:<30} {ds:<10} {base:>8.4f} {add01:>8.4f} {best_ndcg:>8.4f} {best.get('config','?'):<16}")

    print("\n  DONE!")
