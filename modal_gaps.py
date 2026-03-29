"""
A²RAG — Fill remaining gaps: Phase 1 gte-small, statistical significance,
isotropy correlation, quantization for 4 more models.

Usage: modal run --detach modal_gaps.py

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

app = modal.App("a2rag-gaps", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS_6 = [
    ("all-MiniLM-L6-v2", 22, "distilled"),
    ("BAAI/bge-small-en-v1.5", 33, "contrastive"),
    ("all-mpnet-base-v2", 109, "trained-1B-pairs"),
    ("BAAI/bge-base-en-v1.5", 109, "contrastive"),
    ("intfloat/e5-small-v2", 33, "instruction-tuned"),
    ("thenlper/gte-small", 33, "general-text"),
]

DATASETS = ["scifact", "arguana"]


# ═══════════════════════════════════════════════════════════════════
# GAP 1: Phase 1 for gte-small (same ops as modal_17ops_phase1.py)
# ═══════════════════════════════════════════════════════════════════

# (Already completed — gte-small appeared in Phase 2 and 3 but was
#  missing from Phase 1. However, checking the volume shows only 5
#  files in 17ops_phase1. This function fills the gap.)

def op_nlerp(query_embs, target_emb, alpha=0.1):
    import numpy as np
    results = (1 - alpha) * query_embs + alpha * target_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_slerp(query_embs, target_emb, alpha=0.1):
    import numpy as np
    q_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    t_norm = target_emb / np.linalg.norm(target_emb)
    dots = np.clip(q_norm @ t_norm, -1.0, 1.0)
    omegas = np.arccos(dots)
    results = np.zeros_like(query_embs)
    for i in range(len(query_embs)):
        omega = omegas[i]
        if omega < 1e-6:
            results[i] = q_norm[i]
        else:
            sin_omega = np.sin(omega)
            results[i] = (np.sin((1 - alpha) * omega) / sin_omega) * q_norm[i] + \
                          (np.sin(alpha * omega) / sin_omega) * t_norm
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_subtraction(query_embs, exclude_emb):
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_scaled_subtraction(query_embs, exclude_emb, beta=0.5):
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - beta * (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_directional_scaling(query_embs, direction_emb, gamma=1.5):
    import numpy as np
    d_norm = direction_emb / (np.linalg.norm(direction_emb) + 1e-10)
    proj = (query_embs @ d_norm).reshape(-1, 1)
    parallel = proj * d_norm
    orthogonal = query_embs - parallel
    results = orthogonal + gamma * parallel
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)

def op_addition(query_embs, concept_emb, alpha=0.1):
    import numpy as np
    results = query_embs + alpha * concept_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def phase1_gte_small():
    """Fill gap: Phase 1 for gte-small."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    model_name = "thenlper/gte-small"
    print(f"Phase 1 gap fill: {model_name}")
    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    all_results = {"model": model_name, "params_M": 33, "family": "general-text"}

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
            target_text, exclude_text = "clinical medicine and patient outcomes", "methodology and statistical analysis"
        else:
            target_text, exclude_text = "economic policy and market regulation", "moral and ethical reasoning"
        target_emb = model.encode(target_text, normalize_embeddings=True)
        exclude_emb = model.encode(exclude_text, normalize_embeddings=True)

        def eval_ndcg(q, c):
            sims = q @ c.T
            res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
            ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
            return ndcg.get("NDCG@10", 0)

        def jaccard_shift(q1, q2, c, k=10):
            s1, s2 = q1 @ c.T, q2 @ c.T
            j = []
            for i in range(len(q1)):
                t1 = set(np.argsort(s1[i])[::-1][:k])
                t2 = set(np.argsort(s2[i])[::-1][:k])
                j.append(len(t1&t2)/len(t1|t2) if t1|t2 else 1.0)
            return float(np.mean(j))

        baseline = eval_ndcg(query_embs, corpus_embs)
        ds_r = {"baseline_ndcg10": baseline}

        # All Phase 1 operations (same as modal_17ops_phase1.py)
        for alpha in [0.1, 0.2]:
            m = op_nlerp(query_embs, target_emb, alpha)
            ds_r[f"nlerp_a{alpha}"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}
            m2 = op_slerp(query_embs, target_emb, alpha)
            ds_r[f"slerp_a{alpha}"] = {"ndcg10": eval_ndcg(m2, corpus_embs), "delta": round(eval_ndcg(m2, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m2,corpus_embs)}

        m = op_subtraction(query_embs, exclude_emb)
        ds_r["subtraction"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}

        for beta in [0.25, 0.5, 0.75]:
            m = op_scaled_subtraction(query_embs, exclude_emb, beta)
            ds_r[f"scaled_sub_b{beta}"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}

        for gamma in [0.5, 1.5, 2.0]:
            m = op_directional_scaling(query_embs, target_emb, gamma)
            ds_r[f"dir_scale_g{gamma}"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}

        for alpha in [0.1, 0.2]:
            m = op_addition(query_embs, target_emb, alpha)
            ds_r[f"addition_a{alpha}"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}

        # Sequential
        m = op_subtraction(op_nlerp(query_embs, target_emb, 0.1), exclude_emb)
        ds_r["seq_rot_sub"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}
        m = op_nlerp(op_subtraction(query_embs, exclude_emb), target_emb, 0.1)
        ds_r["seq_sub_rot"] = {"ndcg10": eval_ndcg(m, corpus_embs), "delta": round(eval_ndcg(m, corpus_embs)-baseline,4), "jaccard": jaccard_shift(query_embs,m,corpus_embs)}

        all_results[ds_name] = ds_r
        print(f"  {ds_name}: baseline={baseline:.4f}")

    out_path = "/results/17ops_phase1/thenlper_gte-small.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    return all_results


# ═══════════════════════════════════════════════════════════════════
# GAP 2+3: Statistical significance + isotropy correlation for 6 models
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def significance_and_isotropy(model_name: str, params_m: int, family: str):
    """Per-query significance tests + isotropy measurement for one model."""
    import numpy as np
    from scipy import stats
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"Significance + isotropy: {model_name}")
    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()
    all_results = {"model": model_name, "params_M": params_m, "family": family}

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
            exclude_text = "methodology and statistical analysis"
        else:
            target_text = "economic policy and market regulation"
            exclude_text = "moral and ethical reasoning"
        target_emb = model.encode(target_text, normalize_embeddings=True)
        exclude_emb = model.encode(exclude_text, normalize_embeddings=True)

        # Isotropy: mean pairwise cosine of random corpus pairs
        np.random.seed(42)
        n_pairs = 5000
        n = len(corpus_embs)
        ia = np.random.randint(0, n, n_pairs)
        ib = np.random.randint(0, n, n_pairs)
        mask = ia != ib
        ia, ib = ia[mask], ib[mask]
        mean_cos = float(np.sum(corpus_embs[ia] * corpus_embs[ib], axis=1).mean())
        std_cos = float(np.sum(corpus_embs[ia] * corpus_embs[ib], axis=1).std())

        # Per-query nDCG for significance tests
        def per_query_ndcg(q_embs, c_embs):
            sims = q_embs @ c_embs.T
            per_q = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:100]
                per_q_res = {qid: {doc_ids[idx]: float(sims[i, idx]) for idx in top}}
                ndcg, _, _, _ = evaluator.evaluate(
                    {qid: qrels.get(qid, {})}, per_q_res, [10])
                per_q[qid] = ndcg.get("NDCG@10", 0)
            return per_q

        baseline_pq = per_query_ndcg(query_embs, corpus_embs)
        baseline_values = [baseline_pq[qid] for qid in query_ids]
        baseline_mean = float(np.mean(baseline_values))

        ds_r = {
            "n_queries": len(query_ids),
            "n_docs": len(doc_ids),
            "dim": corpus_embs.shape[1],
            "baseline_ndcg10": baseline_mean,
            "isotropy": {"mean_cosine": mean_cos, "std_cosine": std_cos},
        }

        # Test rotation and subtraction
        for op_name, op_fn in [
            ("rotation_0.1", lambda q: op_nlerp(q, target_emb, 0.1)),
            ("rotation_0.2", lambda q: op_nlerp(q, target_emb, 0.2)),
            ("subtraction", lambda q: op_subtraction(q, exclude_emb)),
        ]:
            modified = op_fn(query_embs)
            mod_pq = per_query_ndcg(modified, corpus_embs)
            mod_values = [mod_pq[qid] for qid in query_ids]
            mod_mean = float(np.mean(mod_values))

            # Paired differences
            diffs = [mod_values[i] - baseline_values[i] for i in range(len(query_ids))]
            diffs_arr = np.array(diffs)

            # Tests
            t_stat, t_p = stats.ttest_rel(mod_values, baseline_values)
            try:
                w_stat, w_p = stats.wilcoxon(diffs_arr[diffs_arr != 0])
            except ValueError:
                w_stat, w_p = 0, 1.0

            # Win/tie/loss
            wins = int(np.sum(diffs_arr > 0.001))
            losses = int(np.sum(diffs_arr < -0.001))
            ties = len(diffs) - wins - losses

            ci_low = float(np.mean(diffs) - 1.96 * np.std(diffs) / np.sqrt(len(diffs)))
            ci_high = float(np.mean(diffs) + 1.96 * np.std(diffs) / np.sqrt(len(diffs)))

            ds_r[op_name] = {
                "ndcg10": mod_mean,
                "delta": round(mod_mean - baseline_mean, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "p_ttest": float(t_p),
                "p_wilcoxon": float(w_p),
                "wins": wins, "ties": ties, "losses": losses,
            }
            print(f"  {ds_name} {op_name}: Δ={mod_mean-baseline_mean:+.4f} p={t_p:.4f} W/T/L={wins}/{ties}/{losses}")

        all_results[ds_name] = ds_r

    # Save
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/significance/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    return all_results


# ═══════════════════════════════════════════════════════════════════
# GAP 4: Quantization for 4 additional models
# ═══════════════════════════════════════════════════════════════════

def quantize_int8(embs):
    import numpy as np
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-10)
    quantized = np.round(embs / scales * 127).clip(-127, 127).astype(np.int8)
    return quantized.astype(np.float32) * scales / 127

def quantize_int4(embs):
    import numpy as np
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-10)
    quantized = np.round(embs / scales * 7).clip(-7, 7).astype(np.int8)
    return quantized.astype(np.float32) * scales / 7

def quantize_binary(embs):
    import numpy as np
    binary = np.sign(embs).astype(np.float32)
    norms = np.linalg.norm(binary, axis=1, keepdims=True)
    return binary / np.maximum(norms, 1e-10)

QUANT_METHODS_LITE = {
    "fp32": lambda x: x,
    "int8": quantize_int8,
    "int4": quantize_int4,
    "binary": quantize_binary,
}

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def quantization_extra(model_name: str, dataset_name: str):
    """Quantization benchmark for models not yet tested."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"Quant: {model_name} on {dataset_name}")
    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, f"/tmp/beir-{dataset_name}")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))

    if dataset_name == "scifact":
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

    results = {"model": model_name, "dataset": dataset_name}

    for q_name, q_fn in QUANT_METHODS_LITE.items():
        corpus_q = q_fn(corpus_embs)
        query_q = q_fn(query_embs)
        base = eval_ndcg(query_q, corpus_q)

        rot = op_nlerp(query_q, q_fn(target_emb.reshape(1,-1))[0], 0.1)
        rot_ndcg = eval_ndcg(rot, corpus_q)

        # Angular fidelity
        np.random.seed(42)
        n = min(500, len(corpus_embs))
        idx = np.random.choice(len(corpus_embs), n, replace=False)
        s1 = corpus_embs[idx] @ corpus_embs[idx].T
        s2 = corpus_q[idx] @ corpus_q[idx].T
        mask = np.triu_indices(n, k=1)
        fid = float(np.corrcoef(s1[mask], s2[mask])[0, 1])

        results[q_name] = {
            "baseline_ndcg10": base,
            "rotation_0.1_ndcg10": rot_ndcg,
            "rotation_delta": round(rot_ndcg - base, 4),
            "angular_fidelity": fid,
        }
        print(f"  {q_name}: base={base:.4f} rot={rot_ndcg:.4f} fid={fid:.6f}")

    safe = model_name.replace("/", "_")
    out_path = f"/results/quantization/{safe}_{dataset_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  Filling gaps: Phase1 gte-small + Significance + Quantization")
    print("=" * 70)

    # Gap 1: Phase 1 gte-small
    phase1_gte_small.remote()

    # Gap 2+3: Significance + isotropy for all 6 models (parallel)
    sig_results = list(significance_and_isotropy.map(
        [m[0] for m in MODELS_6],
        [m[1] for m in MODELS_6],
        [m[2] for m in MODELS_6],
    ))

    # Print isotropy correlation
    import numpy as np
    isotropies = []
    deltas = []
    for r in sig_results:
        for ds in DATASETS:
            iso = r[ds]["isotropy"]["mean_cosine"]
            delta = r[ds]["rotation_0.1"]["delta"]
            isotropies.append(iso)
            deltas.append(delta)
            print(f"  {r['model'].split('/')[-1]:<25} {ds:<8} iso={iso:.4f} Δrot={delta:+.4f}")

    iso_arr = np.array(isotropies)
    delta_arr = np.array(deltas)
    from scipy import stats
    r_pearson, p_pearson = stats.pearsonr(iso_arr, delta_arr)
    r_spearman, p_spearman = stats.spearmanr(iso_arr, delta_arr)
    print(f"\n  Isotropy-Efficacy Correlation:")
    print(f"    Pearson:  r={r_pearson:.4f}, p={p_pearson:.4f}")
    print(f"    Spearman: r={r_spearman:.4f}, p={p_spearman:.4f}")

    # Gap 4: Quantization for 4 extra models
    extra_models = ["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5",
                    "intfloat/e5-small-v2", "thenlper/gte-small"]
    q_model_args = []
    q_ds_args = []
    for m in extra_models:
        for d in DATASETS:
            q_model_args.append(m)
            q_ds_args.append(d)

    quant_results = list(quantization_extra.map(q_model_args, q_ds_args))

    print("\n  ALL GAPS FILLED!")
    print("  Download: modal volume get a2rag-results significance/ ./results_modal/significance/")
