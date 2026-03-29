"""
A²RAG Benchmarks on Modal.com — Items 9-12.

Runs all compute-intensive benchmarks on Modal cloud infrastructure,
avoiding MacBook memory crashes. Each model/dataset runs in its own
isolated container with dedicated GPU.

Author: Renato Aparecido Gomes
"""

import modal
import json
import os

# ─── Modal Infrastructure ───────────────────────────────────────────

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

app = modal.App("a2rag-benchmarks", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

# ─── A²RAG Core (inlined to avoid import issues on Modal) ──────────

def rotate_toward(source_emb, target_emb, alpha=0.4):
    import numpy as np
    result = (1 - alpha) * source_emb + alpha * target_emb
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


def subtract_orthogonal(base_emb, exclude_emb):
    import numpy as np
    proj = np.dot(base_emb, exclude_emb) / (np.dot(exclude_emb, exclude_emb) + 1e-10)
    result = base_emb - proj * exclude_emb
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


# ═══════════════════════════════════════════════════════════════════
# ITEM 12: Multi-model benchmark
# ═══════════════════════════════════════════════════════════════════

MODELS = [
    ("all-MiniLM-L6-v2", 22, "distilled"),
    ("BAAI/bge-small-en-v1.5", 33, "contrastive"),
    ("all-mpnet-base-v2", 109, "trained-1B-pairs"),
    ("BAAI/bge-base-en-v1.5", 109, "contrastive"),
    ("intfloat/e5-small-v2", 33, "instruction-tuned"),
    ("thenlper/gte-small", 33, "general-text"),
]

DATASETS_12 = ["scifact", "arguana"]

ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}
SUBTRACTION_CONCEPTS = {
    "scifact": "methodology and statistical analysis",
    "arguana": "economic arguments",
}


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def item12_run_model(model_name: str, params_m: int, family: str):
    """Run Item 12 benchmark for ONE model on both datasets."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir import util
    import time

    print(f"Loading model: {model_name} ({params_m}M, {family})")
    model = SentenceTransformer(model_name)

    model_results = {"model": model_name, "params_M": params_m, "family": family}

    for ds_name in DATASETS_12:
        print(f"\n  Dataset: {ds_name}")

        # Download dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        data_path = util.download_and_unzip(url, "/tmp/beir-data")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                     for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        t0 = time.time()
        corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                             normalize_embeddings=True, show_progress_bar=False))
        query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                            show_progress_bar=False))
        encode_time = time.time() - t0

        evaluator = EvaluateRetrieval()
        result = {
            "model": model_name, "dataset": ds_name,
            "num_docs": len(doc_ids), "num_queries": len(query_ids),
            "embedding_dim": int(corpus_embs.shape[1]),
            "encode_time_s": round(encode_time, 1),
        }

        # Isotropy
        np.random.seed(42)
        n_pairs = 5000
        n = len(corpus_embs)
        idx_a = np.random.randint(0, n, size=n_pairs)
        idx_b = np.random.randint(0, n, size=n_pairs)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        cos_sims = np.sum(corpus_embs[idx_a] * corpus_embs[idx_b], axis=1)
        result["isotropy"] = {
            "mean_cosine": float(cos_sims.mean()),
            "std_cosine": float(cos_sims.std()),
            "median_cosine": float(np.median(cos_sims)),
        }

        # Baseline
        sims = query_embs @ corpus_embs.T
        base_res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            base_res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, map_s, recall, prec = evaluator.evaluate(qrels, base_res, [1, 5, 10])
        result["baseline"] = {
            "ndcg@10": ndcg.get("NDCG@10", 0),
            "ndcg@5": ndcg.get("NDCG@5", 0),
            "map@10": map_s.get("MAP@10", 0),
            "recall@10": recall.get("Recall@10", 0),
        }

        # Rotation α=0.1 and α=0.2
        target_emb = model.encode(ROTATION_TARGETS[ds_name], normalize_embeddings=True)
        for alpha in [0.1, 0.2]:
            rotated = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
            rot_sims = rotated @ corpus_embs.T
            rot_res = {}
            for i, qid in enumerate(query_ids):
                top = np.argsort(rot_sims[i])[::-1][:100]
                rot_res[qid] = {doc_ids[idx]: float(rot_sims[i, idx]) for idx in top}
            ndcg_r, _, _, _ = evaluator.evaluate(qrels, rot_res, [10])
            rot_ndcg = ndcg_r.get("NDCG@10", 0)
            bl = result["baseline"]["ndcg@10"]
            result[f"rotation_{alpha}"] = {
                "ndcg@10": rot_ndcg,
                "delta": round(rot_ndcg - bl, 4),
                "delta_pct": round((rot_ndcg - bl) / bl * 100, 2) if bl > 0 else 0,
            }

        # Subtraction
        concept_emb = model.encode(SUBTRACTION_CONCEPTS[ds_name], normalize_embeddings=True)
        sub = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
        sub_sims = sub @ corpus_embs.T
        sub_res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sub_sims[i])[::-1][:100]
            sub_res[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
        ndcg_s, _, _, _ = evaluator.evaluate(qrels, sub_res, [10])
        sub_ndcg = ndcg_s.get("NDCG@10", 0)
        bl = result["baseline"]["ndcg@10"]
        result["subtraction"] = {
            "ndcg@10": sub_ndcg,
            "delta": round(sub_ndcg - bl, 4),
            "delta_pct": round((sub_ndcg - bl) / bl * 100, 2) if bl > 0 else 0,
        }

        # Projection stats
        proj_norms = np.abs(query_embs @ concept_emb)
        result["projection_stats"] = {
            "mean": float(proj_norms.mean()),
            "std": float(proj_norms.std()),
            "pct_above_0.2": float((proj_norms > 0.2).mean()),
        }

        model_results[ds_name] = result
        bl = result["baseline"]["ndcg@10"]
        rot = result["rotation_0.1"]["ndcg@10"]
        sub_v = result["subtraction"]["ndcg@10"]
        iso = result["isotropy"]["mean_cosine"]
        print(f"    Baseline: {bl:.4f} | Rot 0.1: {rot:.4f} ({result['rotation_0.1']['delta']:+.4f}) "
              f"| Sub: {sub_v:.4f} ({result['subtraction']['delta']:+.4f}) | Isotropy: {iso:.4f}")

    # Save per-model results
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/item12/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"  Saved to {out_path}")

    return model_results


# ═══════════════════════════════════════════════════════════════════
# ITEM 11: Projection threshold analysis
# ═══════════════════════════════════════════════════════════════════

SUBTRACTION_CONCEPTS_11 = {
    "scifact": ["methodology and statistical analysis", "animal model studies",
                "genetic analysis", "clinical trials"],
    "fiqa": ["cryptocurrency", "real estate investment",
             "retirement planning", "tax optimization"],
    "nfcorpus": ["dietary supplements", "weight loss and obesity",
                 "cancer treatment", "pediatric nutrition"],
    "arguana": ["economic arguments", "moral and ethical reasoning",
                "legal precedent", "environmental impact"],
}

BINS = [
    ("very_low", 0.0, 0.1),
    ("low", 0.1, 0.2),
    ("medium", 0.2, 0.3),
    ("high", 0.3, 0.5),
    ("very_high", 0.5, 1.0),
]


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def item11_run_dataset(ds_name: str):
    """Run Item 11 threshold analysis for ONE dataset."""
    import numpy as np
    from scipy import stats
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from pytrec_eval import RelevanceEvaluator

    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Item 11: {ds_name} with {model_name}")

    model = SentenceTransformer(model_name)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    print(f"  Encoding {len(doc_texts)} docs, {len(query_texts)} queries...")
    corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                         normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                        show_progress_bar=False))

    per_query_eval = RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    # Baseline per-query
    base_sims = query_embs @ corpus_embs.T
    base_results = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(base_sims[i])[::-1][:100]
        base_results[qid] = {doc_ids[idx]: float(base_sims[i, idx]) for idx in top}
    base_pq = per_query_eval.evaluate(base_results)

    ds_results = {"dataset": ds_name, "concepts": {}}
    valid_qids = [qid for qid in query_ids if qid in base_pq]

    for concept in SUBTRACTION_CONCEPTS_11[ds_name]:
        concept_emb = model.encode(concept, normalize_embeddings=True)
        projections = np.abs(query_embs @ concept_emb)

        sub_embs = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
        sub_sims = sub_embs @ corpus_embs.T
        sub_results_dict = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sub_sims[i])[::-1][:100]
            sub_results_dict[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
        sub_pq = per_query_eval.evaluate(sub_results_dict)

        concept_result = {
            "projection_distribution": {
                "mean": float(projections.mean()),
                "std": float(projections.std()),
                "min": float(projections.min()),
                "max": float(projections.max()),
                "percentiles": {f"p{p}": float(np.percentile(projections, p))
                                for p in [10, 25, 50, 75, 90]},
            },
            "bins": {},
        }

        print(f"\n  Concept: '{concept}' — proj mean={projections.mean():.3f}")

        for bin_name, lo, hi in BINS:
            bin_qids = []
            for j, qid in enumerate(valid_qids):
                idx = query_ids.index(qid)
                if lo <= projections[idx] < hi:
                    bin_qids.append(qid)

            if len(bin_qids) < 3:
                concept_result["bins"][bin_name] = {
                    "range": [lo, hi], "n_queries": len(bin_qids),
                    "note": "too few queries",
                }
                continue

            base_scores = np.array([base_pq[qid]["ndcg_cut_10"] for qid in bin_qids])
            sub_scores = np.array([sub_pq[qid]["ndcg_cut_10"] for qid in bin_qids])
            deltas = sub_scores - base_scores

            if np.any(deltas != 0):
                t_stat, t_pval = stats.ttest_rel(base_scores, sub_scores)
            else:
                t_stat, t_pval = 0.0, 1.0

            concept_result["bins"][bin_name] = {
                "range": [lo, hi], "n_queries": len(bin_qids),
                "base_ndcg_mean": float(base_scores.mean()),
                "sub_ndcg_mean": float(sub_scores.mean()),
                "delta_mean": float(deltas.mean()),
                "delta_std": float(deltas.std()),
                "t_pvalue": float(t_pval),
                "pct_improved": float((deltas > 0).mean()),
                "pct_degraded": float((deltas < 0).mean()),
                "pct_unchanged": float((deltas == 0).mean()),
            }

            sig = "*" if t_pval < 0.05 else ""
            print(f"    [{lo:.1f}-{hi:.1f}] n={len(bin_qids):>4}  "
                  f"base={base_scores.mean():.4f}  sub={sub_scores.mean():.4f}  "
                  f"Δ={deltas.mean():+.4f}  p={t_pval:.3f}{sig}")

        ds_results["concepts"][concept] = concept_result

    # Control: orthogonal random concepts
    print(f"\n  Control: random orthogonal concepts...")
    np.random.seed(42)
    dim = corpus_embs.shape[1]
    query_centroid = query_embs.mean(axis=0)
    query_centroid = query_centroid / np.linalg.norm(query_centroid)

    control_results = []
    for trial in range(5):
        rand_vec = np.random.randn(dim).astype(np.float32)
        rand_vec = rand_vec - np.dot(rand_vec, query_centroid) * query_centroid
        rand_vec = rand_vec / np.linalg.norm(rand_vec)

        projs = np.abs(query_embs @ rand_vec)
        sub_embs = np.array([subtract_orthogonal(q, rand_vec) for q in query_embs])
        sub_sims = sub_embs @ corpus_embs.T
        sub_res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sub_sims[i])[::-1][:100]
            sub_res[qid] = {doc_ids[idx]: float(sub_sims[i, idx]) for idx in top}
        sub_pq_ctrl = per_query_eval.evaluate(sub_res)

        base_arr = np.array([base_pq[qid]["ndcg_cut_10"] for qid in valid_qids])
        sub_arr = np.array([sub_pq_ctrl[qid]["ndcg_cut_10"] for qid in valid_qids])
        delta = sub_arr - base_arr

        control_results.append({
            "trial": trial,
            "proj_to_centroid": float(abs(np.dot(rand_vec, query_centroid))),
            "mean_proj_to_queries": float(projs.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "pct_changed": float((delta != 0).mean()),
        })
        print(f"    Trial {trial}: Δ={delta.mean():+.5f}  changed={((delta != 0).mean())*100:.1f}%")

    ds_results["control_orthogonal"] = control_results

    # Save
    out_path = f"/results/item11/{ds_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ds_results, f, indent=2)
    vol.commit()
    print(f"  Saved to {out_path}")

    return ds_results


# ═══════════════════════════════════════════════════════════════════
# ITEM 9: Domain validation (drug repurposing + legal analogy)
# ═══════════════════════════════════════════════════════════════════

DRUG_REPURPOSING_QUERIES = {
    "drp_01": "dopamine receptor agonists for motor symptom treatment",
    "drp_02": "levodopa therapy and dyskinesia side effects",
    "drp_03": "alpha-synuclein aggregation and neurodegeneration",
    "drp_04": "deep brain stimulation for movement disorders",
    "drp_05": "mitochondrial dysfunction in dopaminergic neurons",
    "drp_06": "neuroinflammation and microglial activation in brain",
    "drp_07": "oxidative stress and protein misfolding pathology",
    "drp_08": "LRRK2 gene mutation and familial disease risk",
    "drp_09": "gut microbiome dysbiosis and neurological disease",
    "drp_10": "autophagy pathway dysfunction and cell death",
    "drp_11": "tyrosine kinase inhibitors and tumor growth",
    "drp_12": "immune checkpoint blockade for solid tumors",
    "drp_13": "apoptosis resistance mechanisms in cancer cells",
    "drp_14": "angiogenesis inhibition in tumor microenvironment",
    "drp_15": "epigenetic modifications in malignant transformation",
    "drp_16": "insulin resistance and metabolic syndrome",
    "drp_17": "GLP-1 receptor agonists for glucose regulation",
    "drp_18": "pancreatic beta cell dysfunction and apoptosis",
    "drp_19": "inflammatory cytokines in metabolic disease",
    "drp_20": "AMPK pathway activation and energy homeostasis",
}

ROTATION_TARGETS_MEDICAL = {
    "alzheimers": "Alzheimer's disease amyloid beta tau protein neurodegeneration dementia",
    "diabetes": "diabetes mellitus type 2 insulin resistance glucose metabolism",
    "multiple_sclerosis": "multiple sclerosis demyelination autoimmune neuroinflammation",
    "cancer": "cancer tumor oncology malignant neoplasm chemotherapy",
}

LEGAL_QUERIES = {
    "leg_01": "progressive taxation and income redistribution",
    "leg_02": "tax evasion penalties and enforcement mechanisms",
    "leg_03": "constitutional limits on government taxing power",
    "leg_04": "international tax treaties and double taxation",
    "leg_05": "corporate tax optimization and legal avoidance strategies",
    "leg_06": "value added tax collection and compliance",
    "leg_07": "tax exemptions for charitable organizations",
    "leg_08": "fiscal federalism and subnational taxation",
    "leg_09": "retroactive tax legislation and legal certainty",
    "leg_10": "proportionality principle in tax burden distribution",
    "leg_11": "criminal liability and mens rea requirement",
    "leg_12": "proportionality of punishment and sentencing guidelines",
    "leg_13": "burden of proof and presumption of innocence",
    "leg_14": "due process rights in criminal proceedings",
    "leg_15": "judicial review of administrative penalties",
}

ROTATION_TARGETS_LEGAL = {
    "criminal": "criminal law punishment sentencing liability mens rea",
    "consumer": "consumer protection rights warranty liability defective products",
    "administrative": "administrative law government regulation public policy bureaucracy",
    "environmental": "environmental law pollution regulation sustainability climate",
}


@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def item9_run():
    """Run Item 9: domain validation experiments."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from pytrec_eval import RelevanceEvaluator

    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Item 9: Domain Validation with {model_name}")
    model = SentenceTransformer(model_name)

    all_results = {}

    # ── 9A: Drug Repurposing via SciFact ──
    print(f"\n{'='*60}\n  9A: Drug Repurposing (SciFact)\n{'='*60}")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries_sf, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    print(f"  Corpus: {len(doc_ids)} docs")

    corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                         normalize_embeddings=True, show_progress_bar=False))

    drp_ids = list(DRUG_REPURPOSING_QUERIES.keys())
    drp_texts = list(DRUG_REPURPOSING_QUERIES.values())
    drp_embs = np.array(model.encode(drp_texts, normalize_embeddings=True))

    alphas = [0.1, 0.2, 0.3, 0.5]
    drp_results = {"label": "drug_repurposing", "targets": {}}

    baseline_drp = {}
    sims_base = drp_embs @ corpus_embs.T
    for i in range(len(drp_ids)):
        top_idx = np.argsort(sims_base[i])[::-1][:10]
        baseline_drp[i] = [(doc_ids[idx], float(sims_base[i, idx])) for idx in top_idx]

    for target_name, target_text in ROTATION_TARGETS_MEDICAL.items():
        target_emb = model.encode(target_text, normalize_embeddings=True)
        target_data = {"target": target_text, "alphas": {}}

        for alpha in alphas:
            rotated = np.array([rotate_toward(q, target_emb, alpha) for q in drp_embs])
            rot_sims = rotated @ corpus_embs.T

            overlaps = []
            new_docs_counts = []
            examples = []

            for i, qid in enumerate(drp_ids):
                top_idx = np.argsort(rot_sims[i])[::-1][:10]
                rot_docs = [(doc_ids[idx], float(rot_sims[i, idx])) for idx in top_idx]

                base_set = set(d[0] for d in baseline_drp[i])
                rot_set = set(d[0] for d in rot_docs)
                overlap = len(base_set & rot_set) / len(base_set | rot_set) if base_set | rot_set else 1.0
                new_docs = rot_set - base_set
                overlaps.append(overlap)
                new_docs_counts.append(len(new_docs))

                if len(examples) < 3 and len(new_docs) > 0:
                    examples.append({
                        "query": DRUG_REPURPOSING_QUERIES[qid],
                        "baseline_titles": [corpus[d[0]].get("title", "")[:100] for d in baseline_drp[i][:3]],
                        "new_titles": [corpus[did].get("title", "")[:100] for did in list(new_docs)[:3]],
                    })

            target_data["alphas"][f"alpha={alpha}"] = {
                "mean_jaccard": float(np.mean(overlaps)),
                "mean_new_docs": float(np.mean(new_docs_counts)),
                "pct_queries_shifted": float(np.mean([o < 1.0 for o in overlaps])),
                "examples": examples,
            }
            print(f"  {target_name} α={alpha}: Jaccard={np.mean(overlaps):.3f} "
                  f"new_docs={np.mean(new_docs_counts):.1f}")

        drp_results["targets"][target_name] = target_data
    all_results["9A_drug_repurposing"] = drp_results

    # ── 9B: Legal Analogy via ArguAna ──
    print(f"\n{'='*60}\n  9B: Legal Analogy (ArguAna)\n{'='*60}")

    url_ar = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip"
    data_path_ar = util.download_and_unzip(url_ar, "/tmp/beir-data")
    corpus_ar, queries_ar, qrels_ar = GenericDataLoader(data_path_ar).load(split="test")

    doc_ids_ar = list(corpus_ar.keys())
    doc_texts_ar = [(corpus_ar[d].get("title", "") + " " + corpus_ar[d].get("text", "")).strip()
                     for d in doc_ids_ar]
    print(f"  Corpus: {len(doc_ids_ar)} docs")

    corpus_embs_ar = np.array(model.encode(doc_texts_ar, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=False))

    leg_ids = list(LEGAL_QUERIES.keys())
    leg_texts = list(LEGAL_QUERIES.values())
    leg_embs = np.array(model.encode(leg_texts, normalize_embeddings=True))

    leg_results = {"label": "legal_analogy", "targets": {}}

    baseline_leg = {}
    sims_base_leg = leg_embs @ corpus_embs_ar.T
    for i in range(len(leg_ids)):
        top_idx = np.argsort(sims_base_leg[i])[::-1][:10]
        baseline_leg[i] = [(doc_ids_ar[idx], float(sims_base_leg[i, idx])) for idx in top_idx]

    for target_name, target_text in ROTATION_TARGETS_LEGAL.items():
        target_emb = model.encode(target_text, normalize_embeddings=True)
        target_data = {"target": target_text, "alphas": {}}

        for alpha in alphas:
            rotated = np.array([rotate_toward(q, target_emb, alpha) for q in leg_embs])
            rot_sims = rotated @ corpus_embs_ar.T

            overlaps = []
            new_docs_counts = []

            for i, qid in enumerate(leg_ids):
                top_idx = np.argsort(rot_sims[i])[::-1][:10]
                rot_docs = [(doc_ids_ar[idx], float(rot_sims[i, idx])) for idx in top_idx]

                base_set = set(d[0] for d in baseline_leg[i])
                rot_set = set(d[0] for d in rot_docs)
                overlap = len(base_set & rot_set) / len(base_set | rot_set) if base_set | rot_set else 1.0
                overlaps.append(overlap)
                new_docs_counts.append(len(rot_set - base_set))

            target_data["alphas"][f"alpha={alpha}"] = {
                "mean_jaccard": float(np.mean(overlaps)),
                "mean_new_docs": float(np.mean(new_docs_counts)),
                "pct_queries_shifted": float(np.mean([o < 1.0 for o in overlaps])),
            }
            print(f"  {target_name} α={alpha}: Jaccard={np.mean(overlaps):.3f}")

        leg_results["targets"][target_name] = target_data
    all_results["9B_legal_analogy"] = leg_results

    # ── 9C: Cross-domain SciFact with qrels ──
    print(f"\n{'='*60}\n  9C: Cross-domain SciFact (with qrels)\n{'='*60}")

    query_ids_sf = list(queries_sf.keys())
    query_texts_sf = [queries_sf[q] for q in query_ids_sf]
    query_embs_sf = np.array(model.encode(query_texts_sf, normalize_embeddings=True))

    evaluator = RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    base_sims_sf = query_embs_sf @ corpus_embs.T
    base_res_sf = {}
    for i, qid in enumerate(query_ids_sf):
        top = np.argsort(base_sims_sf[i])[::-1][:100]
        base_res_sf[qid] = {doc_ids[idx]: float(base_sims_sf[i, idx]) for idx in top}
    base_pq_sf = evaluator.evaluate(base_res_sf)
    base_mean = float(np.mean([base_pq_sf[qid]["ndcg_cut_10"] for qid in query_ids_sf if qid in base_pq_sf]))

    cross_results = {"baseline_ndcg": round(base_mean, 4), "targets": {}}

    for target_name, target_text in ROTATION_TARGETS_MEDICAL.items():
        target_emb = model.encode(target_text, normalize_embeddings=True)
        for alpha in [0.05, 0.1, 0.2]:
            rotated = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs_sf])
            rot_sims = rotated @ corpus_embs.T
            rot_res = {}
            for i, qid in enumerate(query_ids_sf):
                top = np.argsort(rot_sims[i])[::-1][:100]
                rot_res[qid] = {doc_ids[idx]: float(rot_sims[i, idx]) for idx in top}
            rot_pq = evaluator.evaluate(rot_res)
            rot_mean = float(np.mean([rot_pq[qid]["ndcg_cut_10"] for qid in query_ids_sf if qid in rot_pq]))
            key = f"{target_name}_alpha={alpha}"
            cross_results["targets"][key] = {
                "ndcg": round(rot_mean, 4),
                "delta": round(rot_mean - base_mean, 4),
            }
            print(f"  {target_name} α={alpha}: nDCG={rot_mean:.4f} Δ={rot_mean-base_mean:+.4f}")

    all_results["9C_cross_domain_scifact"] = cross_results

    # Save
    out_path = "/results/item9/domain_validation_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    vol.commit()
    print(f"\n  Saved to {out_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# ITEM 10: Semantic preprocessing (rewrite step runs locally with
#          Ollama; this function handles encoding + evaluation only)
# ═══════════════════════════════════════════════════════════════════

@app.function(gpu="T4", memory=16384, timeout=3600, volumes={"/results": vol})
def item10_evaluate(rewritten_docs_json: str):
    """Run Item 10 evaluation given pre-rewritten docs (from local Ollama).

    Args:
        rewritten_docs_json: JSON string of {doc_id: rewritten_text}
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from beir.retrieval.evaluation import EvaluateRetrieval

    rewritten_texts = json.loads(rewritten_docs_json)
    model_name = "BAAI/bge-small-en-v1.5"

    print(f"Item 10: Semantic preprocessing evaluation")
    print(f"  {len(rewritten_texts)} rewritten docs, model: {model_name}")

    model = SentenceTransformer(model_name)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir-data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # Identify which docs were rewritten
    rewritten_idx = [i for i, did in enumerate(doc_ids) if did in rewritten_texts]
    print(f"  {len(rewritten_idx)} docs found in corpus")

    # Encode everything
    print("  Encoding original corpus...")
    all_corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                             normalize_embeddings=True, show_progress_bar=False))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True,
                                        show_progress_bar=False))

    # Encode rewritten subset
    orig_subset = [doc_texts[i] for i in rewritten_idx]
    rewr_subset = [rewritten_texts.get(doc_ids[i], doc_texts[i]) for i in rewritten_idx]

    orig_embs = np.array(model.encode(orig_subset, batch_size=256,
                                       normalize_embeddings=True, show_progress_bar=False))
    rewr_embs = np.array(model.encode(rewr_subset, batch_size=256,
                                       normalize_embeddings=True, show_progress_bar=False))

    # Build hybrid corpus (replace rewritten docs)
    rewr_full_embs = all_corpus_embs.copy()
    for i, idx in enumerate(rewritten_idx):
        rewr_full_embs[idx] = rewr_embs[i]

    results = {
        "config": {"model": model_name, "n_docs_rewritten": len(rewritten_idx),
                    "total_docs": len(doc_ids)},
    }

    # Embedding similarity
    cos_sims = np.sum(orig_embs * rewr_embs, axis=1)
    results["embedding_similarity"] = {
        "mean_cosine": float(cos_sims.mean()),
        "std_cosine": float(cos_sims.std()),
        "min_cosine": float(cos_sims.min()),
        "max_cosine": float(cos_sims.max()),
    }
    print(f"  Embedding similarity: {cos_sims.mean():.4f} +/- {cos_sims.std():.4f}")

    # Projection analysis
    concepts = {
        "methodology": "methodology and statistical analysis",
        "animal_studies": "animal model studies",
        "genetics": "genetic analysis",
        "rotation_target": "clinical medicine and patient outcomes",
    }
    results["projections"] = {}
    for cname, ctext in concepts.items():
        cemb = model.encode(ctext, normalize_embeddings=True)
        orig_proj = np.abs(orig_embs @ cemb)
        rewr_proj = np.abs(rewr_embs @ cemb)
        results["projections"][cname] = {
            "concept": ctext,
            "original": {"mean": float(orig_proj.mean()), "std": float(orig_proj.std())},
            "rewritten": {"mean": float(rewr_proj.mean()), "std": float(rewr_proj.std())},
            "delta_pct": float((rewr_proj.mean() - orig_proj.mean()) / orig_proj.mean() * 100),
        }
        print(f"  Proj '{cname}': {orig_proj.mean():.4f} → {rewr_proj.mean():.4f}")

    # Retrieval evaluation
    evaluator = EvaluateRetrieval()

    def eval_ndcg(q_embs, c_embs):
        sims = q_embs @ c_embs.T
        res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(sims[i])[::-1][:100]
            res[qid] = {doc_ids[idx]: float(sims[i, idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, res, [10])
        return ndcg.get("NDCG@10", 0)

    baseline_ndcg = eval_ndcg(query_embs, all_corpus_embs)
    hybrid_ndcg = eval_ndcg(query_embs, rewr_full_embs)
    results["retrieval"] = {
        "baseline_ndcg10": baseline_ndcg,
        "hybrid_ndcg10": hybrid_ndcg,
        "delta": round(hybrid_ndcg - baseline_ndcg, 4),
    }
    print(f"  Baseline nDCG@10: {baseline_ndcg:.4f}")
    print(f"  Hybrid nDCG@10:   {hybrid_ndcg:.4f} ({hybrid_ndcg - baseline_ndcg:+.4f})")

    # Rotation + subtraction on hybrid
    target_emb = model.encode("clinical medicine and patient outcomes", normalize_embeddings=True)
    rotated_q = np.array([rotate_toward(q, target_emb, 0.1) for q in query_embs])
    rot_orig = eval_ndcg(rotated_q, all_corpus_embs)
    rot_hybrid = eval_ndcg(rotated_q, rewr_full_embs)
    results["rotation_0.1"] = {
        "original": rot_orig, "hybrid": rot_hybrid,
        "delta_orig": round(rot_orig - baseline_ndcg, 4),
        "delta_hybrid": round(rot_hybrid - baseline_ndcg, 4),
    }
    print(f"  Rotation: orig={rot_orig:.4f} hybrid={rot_hybrid:.4f}")

    # Isotropy
    np.random.seed(42)
    n_pairs = 3000
    n = len(rewritten_idx)
    if n > 1:
        ia = np.random.randint(0, n, n_pairs)
        ib = np.random.randint(0, n, n_pairs)
        mask = ia != ib
        ia, ib = ia[mask], ib[mask]
        orig_iso = float(np.sum(orig_embs[ia] * orig_embs[ib], axis=1).mean())
        rewr_iso = float(np.sum(rewr_embs[ia] * rewr_embs[ib], axis=1).mean())
        results["isotropy"] = {"original": orig_iso, "rewritten": rewr_iso}
        print(f"  Isotropy: orig={orig_iso:.4f} → rewr={rewr_iso:.4f}")

    # Save
    out_path = "/results/item10/preprocessing_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"  Saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def run_all():
    """Run items 12, 11, and 9 in parallel (item 10 needs local Ollama first)."""
    import numpy as np

    # ── Item 12: 6 models in parallel ──
    print("\n" + "="*70)
    print("  ITEM 12: Multi-model benchmark (6 models in PARALLEL)")
    print("="*70)

    results_12 = list(item12_run_model.map(
        [m[0] for m in MODELS],
        [m[1] for m in MODELS],
        [m[2] for m in MODELS],
    ))

    merged = {r["model"]: r for r in results_12}

    print(f"\n{'='*100}")
    print(f"  SUMMARY — Item 12")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Params':>6} {'Family':<18} {'Dataset':<10} "
          f"{'Baseline':>8} {'Rot 0.1':>8} {'Rot 0.2':>8} {'Sub':>8} {'Isotropy':>8}")
    print("-" * 110)

    for model_name, data in merged.items():
        for ds in DATASETS_12:
            r = data[ds]
            short_name = model_name.split("/")[-1]
            print(f"{short_name:<30} {data['params_M']:>4}M  {data['family']:<18} {ds:<10} "
                  f"{r['baseline']['ndcg@10']:>8.4f} {r['rotation_0.1']['ndcg@10']:>8.4f} "
                  f"{r['rotation_0.2']['ndcg@10']:>8.4f} {r['subtraction']['ndcg@10']:>8.4f} "
                  f"{r['isotropy']['mean_cosine']:>8.4f}")

    # ── Item 11: 4 datasets in parallel ──
    print("\n" + "="*70)
    print("  ITEM 11: Projection threshold (4 datasets in PARALLEL)")
    print("="*70)

    datasets_11 = ["scifact", "fiqa", "nfcorpus", "arguana"]
    results_11 = list(item11_run_dataset.map(datasets_11))

    merged_11 = {r["dataset"]: r for r in results_11}

    print(f"\n  Key finding: |proj| < 0.2 → safe subtraction?")
    for ds, data in merged_11.items():
        print(f"\n  {ds}:")
        for concept, cdata in data["concepts"].items():
            low_bins = [b["delta_mean"] for bn, b in cdata["bins"].items()
                       if bn in ("very_low", "low") and "delta_mean" in b]
            high_bins = [b["delta_mean"] for bn, b in cdata["bins"].items()
                        if bn not in ("very_low", "low") and "delta_mean" in b]
            low_avg = np.mean(low_bins) if low_bins else float('nan')
            high_avg = np.mean(high_bins) if high_bins else float('nan')
            print(f"    '{concept[:30]}': low Δ={low_avg:+.4f}  high Δ={high_avg:+.4f}")

    # ── Item 9: single container ──
    print("\n" + "="*70)
    print("  ITEM 9: Domain validation")
    print("="*70)
    results_9 = item9_run.remote()

    print("\n\n" + "="*70)
    print("  ALL DONE! Results saved to Modal Volume 'a2rag-results'")
    print("  Download with: modal volume get a2rag-results / ./results_modal/")
    print("="*70)


@app.local_entrypoint()
def run_item10(rewritten_docs_path: str):
    """Run Item 10 evaluation after local Ollama rewriting.

    Usage: modal run modal_benchmarks.py::run_item10 --rewritten-docs-path results/item10_preprocessing/rewritten_docs.json
    """
    with open(rewritten_docs_path) as f:
        rewritten_json = f.read()

    print("  ITEM 10: Semantic preprocessing evaluation")
    results_10 = item10_evaluate.remote(rewritten_json)

    print(f"\n  Baseline: {results_10['retrieval']['baseline_ndcg10']:.4f}")
    print(f"  Hybrid:   {results_10['retrieval']['hybrid_ndcg10']:.4f} ({results_10['retrieval']['delta']:+.4f})")
    print(f"\n  Results saved to Modal Volume 'a2rag-results'")
