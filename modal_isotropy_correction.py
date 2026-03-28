"""
A2RAG — Isotropy Correction + Algebraic Operations

FW6 mostrou que PCA whitening cru destroi o baseline (e5: 0.68 -> 0.07).
Abordagem cirurgica: 3 estrategias de correcao de isotropy que preservam o baseline.

Estrategias:
1. Mean-centering ("All-but-the-Top"): subtrair a media do corpus dos embeddings.
   A causa principal de anisotropy e uma direcao media dominante.
2. Partial whitening: blend entre original e whitened: emb' = (1-gamma)*emb + gamma*whiten(emb)
3. Top-k removal: remover as top-k componentes principais (que capturam anisotropy, nao semantica)

Para cada estrategia: medir baseline nDCG (deve manter), isotropy (deve melhorar),
e rotation delta (deve ficar menos negativo ou positivo).

Usage: modal run modal_isotropy_correction.py

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

app = modal.App("a2rag-isotropy-correction", image=image)
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


def normalize_rows(X):
    import numpy as np
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def normalize_vec(v):
    import numpy as np
    n = np.linalg.norm(v)
    return v / max(n, 1e-10)


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def run_isotropy_correction(model_name: str, params_m: int, family: str):
    """Test isotropy correction strategies for ONE model on both datasets."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*60}")
    print(f"  Isotropy Correction: {model_name} ({params_m}M, {family})")
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

        # Encode (normalized)
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)

        def measure_isotropy(embs):
            np.random.seed(42)
            n = min(3000, len(embs))
            ia = np.random.randint(0, len(embs), n)
            ib = np.random.randint(0, len(embs), n)
            mask = ia != ib
            return float(np.mean(np.sum(embs[ia[mask]] * embs[ib[mask]], axis=1)))

        def eval_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            results = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                results[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, results, [1, 5, 10])
            return ndcg

        def op_addition(q, target, alpha):
            results = q + alpha * target
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            return results / np.maximum(norms, 1e-10)

        # === BASELINE (no correction) ===
        iso_orig = measure_isotropy(corpus_embs)
        ndcg_base = eval_ndcg(query_embs, corpus_embs)
        base_10 = ndcg_base.get("NDCG@10", 0)
        print(f"    Original: iso={iso_orig:.3f} base={base_10:.4f}")

        ds_results = {
            "original": {
                "isotropy": round(iso_orig, 4),
                "baseline": {k: round(v, 4) for k, v in ndcg_base.items()},
            }
        }

        # Test rotation deltas for original
        for alpha in ALPHAS:
            q_rot = op_addition(query_embs, target_emb, alpha)
            ndcg_rot = eval_ndcg(q_rot, corpus_embs)
            delta = ndcg_rot.get("NDCG@10", 0) - base_10
            ds_results["original"][f"add_delta_a{alpha}"] = round(delta, 4)

        print(f"    Original rotation deltas: {[ds_results['original'].get(f'add_delta_a{a}', 0) for a in ALPHAS]}")

        # === STRATEGY 1: Mean-centering ===
        corpus_mean = corpus_embs.mean(axis=0)
        corpus_mc = normalize_rows(corpus_embs - corpus_mean)
        query_mc = normalize_rows(query_embs - corpus_mean)
        target_mc = normalize_vec(target_emb - corpus_mean)

        iso_mc = measure_isotropy(corpus_mc)
        ndcg_mc = eval_ndcg(query_mc, corpus_mc)
        mc_10 = ndcg_mc.get("NDCG@10", 0)
        print(f"    Mean-center: iso={iso_mc:.3f} base={mc_10:.4f} (Δ={mc_10 - base_10:+.4f})")

        mc_results = {
            "isotropy": round(iso_mc, 4),
            "baseline": {k: round(v, 4) for k, v in ndcg_mc.items()},
            "baseline_delta": round(mc_10 - base_10, 4),
        }
        for alpha in ALPHAS:
            q_rot = op_addition(query_mc, target_mc, alpha)
            ndcg_rot = eval_ndcg(q_rot, corpus_mc)
            delta = ndcg_rot.get("NDCG@10", 0) - mc_10
            mc_results[f"add_delta_a{alpha}"] = round(delta, 4)

        print(f"    Mean-center rotation deltas: {[mc_results.get(f'add_delta_a{a}', 0) for a in ALPHAS]}")
        ds_results["mean_centering"] = mc_results

        # === STRATEGY 2: Partial whitening (blend) ===
        for gamma in [0.1, 0.25, 0.5]:
            pca = PCA(whiten=True)
            corpus_w_full = pca.fit_transform(corpus_embs)
            query_w_full = pca.transform(query_embs)
            target_w_full = pca.transform(target_emb.reshape(1, -1))[0]

            # Blend: (1-gamma)*original + gamma*whitened
            corpus_pw = normalize_rows((1 - gamma) * corpus_embs + gamma * corpus_w_full)
            query_pw = normalize_rows((1 - gamma) * query_embs + gamma * query_w_full)
            target_pw = normalize_vec((1 - gamma) * target_emb + gamma * target_w_full)

            iso_pw = measure_isotropy(corpus_pw)
            ndcg_pw = eval_ndcg(query_pw, corpus_pw)
            pw_10 = ndcg_pw.get("NDCG@10", 0)
            print(f"    Partial-whiten γ={gamma}: iso={iso_pw:.3f} base={pw_10:.4f} (Δ={pw_10 - base_10:+.4f})")

            pw_results = {
                "gamma": gamma,
                "isotropy": round(iso_pw, 4),
                "baseline": {k: round(v, 4) for k, v in ndcg_pw.items()},
                "baseline_delta": round(pw_10 - base_10, 4),
            }
            for alpha in ALPHAS:
                q_rot = op_addition(query_pw, target_pw, alpha)
                ndcg_rot = eval_ndcg(q_rot, corpus_pw)
                delta = ndcg_rot.get("NDCG@10", 0) - pw_10
                pw_results[f"add_delta_a{alpha}"] = round(delta, 4)

            print(f"    Partial-whiten γ={gamma} rotation deltas: {[pw_results.get(f'add_delta_a{a}', 0) for a in ALPHAS]}")
            ds_results[f"partial_whiten_g{gamma}"] = pw_results

        # === STRATEGY 3: Top-k principal component removal ===
        for k in [1, 3, 5]:
            pca = PCA()
            pca.fit(corpus_embs)
            # Remove top-k components
            components_to_remove = pca.components_[:k]  # (k, dim)
            # Project out: emb' = emb - sum(proj onto each component)
            corpus_tk = corpus_embs.copy()
            query_tk = query_embs.copy()
            target_tk = target_emb.copy()
            for comp in components_to_remove:
                corpus_tk -= (corpus_tk @ comp).reshape(-1, 1) * comp
                query_tk -= (query_tk @ comp).reshape(-1, 1) * comp
                target_tk -= np.dot(target_tk, comp) * comp

            corpus_tk = normalize_rows(corpus_tk)
            query_tk = normalize_rows(query_tk)
            target_tk = normalize_vec(target_tk)

            iso_tk = measure_isotropy(corpus_tk)
            ndcg_tk = eval_ndcg(query_tk, corpus_tk)
            tk_10 = ndcg_tk.get("NDCG@10", 0)
            print(f"    Top-{k} removal: iso={iso_tk:.3f} base={tk_10:.4f} (Δ={tk_10 - base_10:+.4f})")

            tk_results = {
                "k": k,
                "isotropy": round(iso_tk, 4),
                "baseline": {k_metric: round(v, 4) for k_metric, v in ndcg_tk.items()},
                "baseline_delta": round(tk_10 - base_10, 4),
            }
            for alpha in ALPHAS:
                q_rot = op_addition(query_tk, target_tk, alpha)
                ndcg_rot = eval_ndcg(q_rot, corpus_tk)
                delta = ndcg_rot.get("NDCG@10", 0) - tk_10
                tk_results[f"add_delta_a{alpha}"] = round(delta, 4)

            print(f"    Top-{k} removal rotation deltas: {[tk_results.get(f'add_delta_a{a}', 0) for a in ALPHAS]}")
            ds_results[f"topk_removal_k{k}"] = tk_results

        model_results[ds_name] = ds_results

    safe_name = model_name.replace("/", "_")
    out_path = f"/results/isotropy_correction/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Isotropy Correction + Algebraic Operations")
    print("  6 models x 2 datasets x 7 strategies x 4 alphas")
    print("=" * 70)

    all_results = list(run_isotropy_correction.starmap(
        [(name, params, fam) for name, params, fam in MODELS]
    ))

    # Summary: best strategy per model
    print("\n" + "=" * 70)
    print("  SUMMARY: Rotation delta @ alpha=0.1 (original vs best correction)")
    print("=" * 70)
    print(f"  {'Model':<30} {'DS':<10} {'Orig_iso':>8} {'Orig_Δ':>8} {'Best_strat':<22} {'Corr_iso':>8} {'Corr_Δ':>8} {'Base_Δ':>8}")
    print("  " + "-" * 104)

    strategies = ["mean_centering", "partial_whiten_g0.1", "partial_whiten_g0.25",
                  "partial_whiten_g0.5", "topk_removal_k1", "topk_removal_k3", "topk_removal_k5"]

    for r in all_results:
        for ds in DATASETS:
            if ds not in r:
                continue
            orig = r[ds]["original"]
            orig_delta = orig.get("add_delta_a0.1", 0)
            orig_iso = orig.get("isotropy", 0)

            best_strat = "original"
            best_delta = orig_delta
            best_iso = orig_iso
            best_base_delta = 0

            for strat in strategies:
                if strat not in r[ds]:
                    continue
                s = r[ds][strat]
                d = s.get("add_delta_a0.1", 0)
                bd = s.get("baseline_delta", 0)
                # Best = highest rotation delta with baseline not degraded more than -0.02
                if bd > -0.02 and d > best_delta:
                    best_delta = d
                    best_strat = strat
                    best_iso = s.get("isotropy", 0)
                    best_base_delta = bd

            print(f"  {r['model']:<30} {ds:<10} {orig_iso:>8.3f} {orig_delta:>+8.4f} {best_strat:<22} {best_iso:>8.3f} {best_delta:>+8.4f} {best_base_delta:>+8.4f}")

    print("\n  DONE!")
