"""
A²RAG — Quantization Robustness Benchmark.

Tests whether A²RAG algebraic operations survive aggressive quantization.
Validates the Future Work item from the paper: "Robustness under quantization".

Protocol:
1. Generate full-precision (FP32) embeddings for SciFact + ArguAna
2. Quantize embeddings to int8, int4, 3-bit, and binary (1-bit sign)
3. Apply rotation (α=0.1, 0.2) and subtraction on each precision level
4. Measure nDCG@10 degradation vs full-precision baseline
5. Measure angular fidelity: does the rotation angle survive quantization?

Key hypothesis (from PolarQuant): angular structure is resilient to quantization,
so A²RAG operations (which are angular) should be robust.

Models: MiniLM (best for A²RAG) + BGE-small (worst for A²RAG) = contrastive pair
Datasets: SciFact + ArguAna

Usage: modal run --detach modal_quantization_robustness.py

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

app = modal.App("a2rag-quantization", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)


# ═══════════════════════════════════════════════════════════════════
# QUANTIZATION METHODS
# ═══════════════════════════════════════════════════════════════════

def quantize_int8(embs):
    """Symmetric int8 quantization: scale to [-127, 127], round, dequantize."""
    import numpy as np
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-10)
    quantized = np.round(embs / scales * 127).clip(-127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scales / 127
    return dequantized


def quantize_int4(embs):
    """Symmetric int4 quantization: scale to [-7, 7], round, dequantize."""
    import numpy as np
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-10)
    quantized = np.round(embs / scales * 7).clip(-7, 7).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scales / 7
    return dequantized


def quantize_3bit(embs):
    """Symmetric 3-bit quantization: scale to [-3, 3], round, dequantize."""
    import numpy as np
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-10)
    quantized = np.round(embs / scales * 3).clip(-3, 3).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scales / 3
    return dequantized


def quantize_binary(embs):
    """Binary (1-bit) quantization: sign projection (QJL-inspired)."""
    import numpy as np
    # Random projection for better binary codes (like QJL)
    np.random.seed(42)
    dim = embs.shape[1]
    # Use sign of original embeddings (simpler, but effective)
    binary = np.sign(embs).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(binary, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return binary / norms


def quantize_polar_int8(embs):
    """PolarQuant-inspired: convert to polar (magnitude + angles), quantize angles in int8."""
    import numpy as np
    # Compute norms (magnitudes)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    # Normalize to unit sphere
    unit = embs / norms
    # Quantize the unit vectors (angular components) in int8
    quantized_unit = quantize_int8(unit)
    # Re-normalize after quantization to stay on sphere
    q_norms = np.linalg.norm(quantized_unit, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-10)
    quantized_unit = quantized_unit / q_norms
    # Restore magnitude (keep full precision magnitude)
    return quantized_unit * norms


QUANT_METHODS = {
    "fp32": lambda x: x,  # no quantization (baseline)
    "int8": quantize_int8,
    "int4": quantize_int4,
    "3bit": quantize_3bit,
    "binary": quantize_binary,
    "polar_int8": quantize_polar_int8,
}


# ═══════════════════════════════════════════════════════════════════
# OPERATIONS (same as main benchmark)
# ═══════════════════════════════════════════════════════════════════

def op_rotate(query_embs, target_emb, alpha=0.1):
    import numpy as np
    results = (1 - alpha) * query_embs + alpha * target_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


def op_subtract(query_embs, exclude_emb):
    import numpy as np
    proj = (query_embs @ exclude_emb).reshape(-1, 1)
    denom = np.dot(exclude_emb, exclude_emb) + 1e-10
    results = query_embs - (proj / denom) * exclude_emb
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    return results / np.maximum(norms, 1e-10)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def benchmark_quantization(model_name: str, dataset_name: str):
    """Test all quantization levels × operations for one model+dataset."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    print(f"\n{'='*70}")
    print(f"  Quantization Robustness: {model_name} on {dataset_name}")
    print(f"{'='*70}")

    model = SentenceTransformer(model_name)
    evaluator = EvaluateRetrieval()

    # Load dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, f"/tmp/beir-{dataset_name}")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Encode at full precision
    print(f"  Encoding {len(doc_texts)} docs + {len(query_texts)} queries (FP32)...")
    corpus_embs_fp32 = np.array(model.encode(doc_texts, batch_size=256,
                                              normalize_embeddings=True, show_progress_bar=False))
    query_embs_fp32 = np.array(model.encode(query_texts, normalize_embeddings=True,
                                             show_progress_bar=False))

    # Concept embeddings
    if dataset_name == "scifact":
        target_text = "clinical medicine and patient outcomes"
        exclude_text = "methodology and statistical analysis"
    else:
        target_text = "economic policy and market regulation"
        exclude_text = "moral and ethical reasoning"

    target_emb = model.encode(target_text, normalize_embeddings=True)
    exclude_emb = model.encode(exclude_text, normalize_embeddings=True)

    def eval_ndcg(q_embs, c_embs):
        sims = q_embs @ c_embs.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
        return ndcg.get("NDCG@10", 0)

    def angular_fidelity(embs_orig, embs_quant):
        """Measure how well pairwise angles are preserved after quantization."""
        np.random.seed(42)
        n = min(500, len(embs_orig))
        idx = np.random.choice(len(embs_orig), n, replace=False)
        sims_orig = embs_orig[idx] @ embs_orig[idx].T
        sims_quant = embs_quant[idx] @ embs_quant[idx].T
        # Correlation of upper triangle
        mask = np.triu_indices(n, k=1)
        corr = np.corrcoef(sims_orig[mask], sims_quant[mask])[0, 1]
        mae = float(np.mean(np.abs(sims_orig[mask] - sims_quant[mask])))
        return {"pearson_r": float(corr), "mae_cosine": mae}

    def rotation_angle_fidelity(q_fp32, q_quant, target, alpha=0.1):
        """Check if the rotation angle is preserved after quantization."""
        rot_fp32 = op_rotate(q_fp32, target, alpha)
        rot_quant = op_rotate(q_quant, target, alpha)
        # Angle between fp32-rotated and quant-rotated
        dots = np.sum(rot_fp32 * rot_quant, axis=1)
        angles_deg = np.degrees(np.arccos(np.clip(dots, -1, 1)))
        return {
            "mean_angle_deviation_deg": float(angles_deg.mean()),
            "max_angle_deviation_deg": float(angles_deg.max()),
            "std_angle_deviation_deg": float(angles_deg.std()),
        }

    results = {
        "model": model_name, "dataset": dataset_name,
        "n_docs": len(doc_ids), "n_queries": len(query_ids),
        "dim": corpus_embs_fp32.shape[1],
    }

    # Test each quantization level
    for q_name, q_fn in QUANT_METHODS.items():
        print(f"\n  --- {q_name} ---")

        # Quantize corpus and queries
        corpus_q = q_fn(corpus_embs_fp32)
        query_q = q_fn(query_embs_fp32)

        # Angular fidelity
        fidelity = angular_fidelity(corpus_embs_fp32, corpus_q)
        print(f"    Angular fidelity: r={fidelity['pearson_r']:.6f}, MAE={fidelity['mae_cosine']:.6f}")

        # Baseline retrieval at this precision
        base_ndcg = eval_ndcg(query_q, corpus_q)
        print(f"    Baseline nDCG@10: {base_ndcg:.4f}")

        q_results = {
            "angular_fidelity": fidelity,
            "baseline_ndcg10": base_ndcg,
        }

        # Rotation at this precision
        for alpha in [0.1, 0.2]:
            # Method A: quantize first, then rotate (realistic deployment)
            rot_post = op_rotate(query_q, q_fn(target_emb.reshape(1, -1))[0], alpha)
            ndcg_post = eval_ndcg(rot_post, corpus_q)

            # Method B: rotate first, then quantize (ideal case)
            rot_pre = op_rotate(query_embs_fp32, target_emb, alpha)
            rot_pre_q = q_fn(rot_pre)
            ndcg_pre = eval_ndcg(rot_pre_q, corpus_q)

            # Angle fidelity
            angle_fid = rotation_angle_fidelity(query_embs_fp32, query_q, target_emb, alpha)

            q_results[f"rotation_a{alpha}"] = {
                "quant_then_rotate": {"ndcg10": ndcg_post, "delta_vs_base": round(ndcg_post - base_ndcg, 4)},
                "rotate_then_quant": {"ndcg10": ndcg_pre, "delta_vs_base": round(ndcg_pre - base_ndcg, 4)},
                "angle_fidelity": angle_fid,
            }
            print(f"    Rot α={alpha}: Q→R={ndcg_post:.4f} R→Q={ndcg_pre:.4f} "
                  f"angle_dev={angle_fid['mean_angle_deviation_deg']:.2f}°")

        # Subtraction at this precision
        sub_post = op_subtract(query_q, q_fn(exclude_emb.reshape(1, -1))[0])
        ndcg_sub = eval_ndcg(sub_post, corpus_q)
        q_results["subtraction"] = {
            "ndcg10": ndcg_sub, "delta_vs_base": round(ndcg_sub - base_ndcg, 4),
        }
        print(f"    Subtraction: nDCG={ndcg_sub:.4f} Δ={ndcg_sub-base_ndcg:+.4f}")

        # Combined: rotate + subtract
        combined = op_subtract(op_rotate(query_q, q_fn(target_emb.reshape(1, -1))[0], 0.1),
                               q_fn(exclude_emb.reshape(1, -1))[0])
        ndcg_comb = eval_ndcg(combined, corpus_q)
        q_results["combined_rot_sub"] = {
            "ndcg10": ndcg_comb, "delta_vs_base": round(ndcg_comb - base_ndcg, 4),
        }
        print(f"    Combined R+S: nDCG={ndcg_comb:.4f} Δ={ndcg_comb-base_ndcg:+.4f}")

        results[q_name] = q_results

    # Compute summary: relative degradation vs FP32
    fp32_base = results["fp32"]["baseline_ndcg10"]
    fp32_rot01 = results["fp32"]["rotation_a0.1"]["quant_then_rotate"]["ndcg10"]
    fp32_sub = results["fp32"]["subtraction"]["ndcg10"]

    summary = {}
    for q_name in QUANT_METHODS:
        if q_name == "fp32":
            continue
        q_base = results[q_name]["baseline_ndcg10"]
        q_rot = results[q_name]["rotation_a0.1"]["quant_then_rotate"]["ndcg10"]
        q_sub = results[q_name]["subtraction"]["ndcg10"]

        # Relative degradation: how much does quantization hurt the OPERATION?
        # (not just the baseline, but the delta from the operation)
        fp32_rot_delta = fp32_rot01 - fp32_base
        q_rot_delta = q_rot - q_base

        summary[q_name] = {
            "baseline_degradation": round(q_base - fp32_base, 4),
            "rotation_delta_fp32": round(fp32_rot_delta, 4),
            "rotation_delta_quant": round(q_rot_delta, 4),
            "rotation_preserved": abs(q_rot_delta - fp32_rot_delta) < 0.01,
            "subtraction_degradation": round(q_sub - fp32_sub, 4),
        }

    results["summary"] = summary

    # Save
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/quantization/{safe_name}_{dataset_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"\n  Saved to {out_path}")

    return results


@app.local_entrypoint()
def main():
    """Run quantization benchmark: 2 models × 2 datasets = 4 containers in parallel."""
    print("=" * 70)
    print("  A²RAG Quantization Robustness Benchmark")
    print("  2 models × 2 datasets × 6 quant levels × 4 operations")
    print("=" * 70)

    models = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
    datasets = ["scifact", "arguana"]

    # Create all (model, dataset) pairs
    model_args = []
    ds_args = []
    for m in models:
        for d in datasets:
            model_args.append(m)
            ds_args.append(d)

    results = list(benchmark_quantization.map(model_args, ds_args))

    # Summary table
    print(f"\n{'='*120}")
    print(f"  QUANTIZATION ROBUSTNESS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Model':<25} {'Dataset':<10} {'Quant':<12} {'Base':>7} {'Rot01':>7} "
          f"{'Sub':>7} {'AngFid':>8} {'RotΔ':>7} {'Preserved':>9}")
    print("-" * 120)

    for r in results:
        model_short = r["model"].split("/")[-1][:24]
        ds = r["dataset"]
        for q_name in ["fp32", "int8", "polar_int8", "int4", "3bit", "binary"]:
            qd = r[q_name]
            base = qd["baseline_ndcg10"]
            rot = qd["rotation_a0.1"]["quant_then_rotate"]["ndcg10"]
            sub = qd["subtraction"]["ndcg10"]
            fid = qd["angular_fidelity"]["pearson_r"]
            rot_delta = rot - base
            preserved = "✓" if q_name == "fp32" else ("✓" if r.get("summary", {}).get(q_name, {}).get("rotation_preserved", False) else "✗")
            print(f"{model_short:<25} {ds:<10} {q_name:<12} {base:>7.4f} {rot:>7.4f} "
                  f"{sub:>7.4f} {fid:>8.6f} {rot_delta:>+7.4f} {preserved:>9}")

    print(f"\n  Key: AngFid = Pearson correlation of pairwise cosines (1.0 = perfect)")
    print(f"       RotΔ = nDCG change from rotation (should match FP32 sign/magnitude)")
    print(f"       Preserved = rotation effect preserved within ±0.01 of FP32")
    print(f"\n  Download: modal volume get a2rag-results quantization/ ./results_modal/quantization/")
