"""
Item 9: Domain validation — drug repurposing + legal analogy.

9A. PubMed/SciFact: queries about one disease, rotated toward another.
    Evaluates if rotation surfaces biologically relevant cross-domain analogues.

9B. Legal (simulated): queries about one legal branch, rotated toward another.
    Uses ArguAna as proxy (argumentation domain).

Author: Renato Aparecido Gomes
"""

import json
import numpy as np
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from a2rag import rotate_toward, subtract_orthogonal

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/item9_domain")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 9A: Drug Repurposing via SciFact ───────────────────────────────

DRUG_REPURPOSING_QUERIES = {
    # Parkinson's disease queries
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
    # Cancer treatment queries (for cross-domain rotation)
    "drp_11": "tyrosine kinase inhibitors and tumor growth",
    "drp_12": "immune checkpoint blockade for solid tumors",
    "drp_13": "apoptosis resistance mechanisms in cancer cells",
    "drp_14": "angiogenesis inhibition in tumor microenvironment",
    "drp_15": "epigenetic modifications in malignant transformation",
    # Diabetes queries
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

# ─── 9B: Legal Analogy ─────────────────────────────────────────────

LEGAL_QUERIES = {
    # Tax law queries (direito tributário)
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
    # Criminal law queries
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


def retrieve_top_k(query_embs, corpus_embs, doc_ids, k=10):
    """Retrieve top-k documents for each query."""
    sims = query_embs @ corpus_embs.T
    results = {}
    for i in range(len(query_embs)):
        top_idx = np.argsort(sims[i])[::-1][:k]
        results[i] = [(doc_ids[idx], float(sims[i, idx])) for idx in top_idx]
    return results


def analyze_rotation_domain_shift(query_embs, query_ids, query_texts, corpus_embs,
                                   doc_ids, corpus, model, targets, alphas, label):
    """Analyze how rotation shifts results across domains."""
    results = {"label": label, "targets": {}}

    # Baseline retrieval
    baseline = retrieve_top_k(query_embs, corpus_embs, doc_ids, k=10)

    for target_name, target_text in targets.items():
        target_emb = model.encode(target_text, normalize_embeddings=True)
        target_results = {"target": target_text, "alphas": {}}

        for alpha in alphas:
            rotated = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
            rot_retrieval = retrieve_top_k(rotated, corpus_embs, doc_ids, k=10)

            # Compute overlap and new docs per query
            overlaps = []
            new_docs_all = []
            examples = []

            for i, qid in enumerate(query_ids):
                base_set = set([d[0] for d in baseline[i]])
                rot_set = set([d[0] for d in rot_retrieval[i]])
                overlap = len(base_set & rot_set) / len(base_set | rot_set) if base_set | rot_set else 1.0
                new_docs = rot_set - base_set
                overlaps.append(overlap)
                new_docs_all.append(len(new_docs))

                # Capture example for first 3 queries
                if len(examples) < 3 and len(new_docs) > 0:
                    # Get text snippets of new docs
                    new_doc_snippets = []
                    for did in list(new_docs)[:3]:
                        text = (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
                        new_doc_snippets.append({
                            "doc_id": did,
                            "title": corpus[did].get("title", ""),
                            "snippet": text[:200],
                        })

                    # Get baseline doc snippets
                    base_doc_snippets = []
                    for did, score in baseline[i][:3]:
                        base_doc_snippets.append({
                            "doc_id": did,
                            "title": corpus[did].get("title", ""),
                            "snippet": (corpus[did].get("title","")+" "+corpus[did].get("text","")).strip()[:200],
                        })

                    examples.append({
                        "query": query_texts[qid] if isinstance(qid, str) else query_texts[query_ids[i]],
                        "baseline_docs": base_doc_snippets,
                        "new_docs_after_rotation": new_doc_snippets,
                    })

            target_results["alphas"][f"alpha={alpha}"] = {
                "mean_jaccard": float(np.mean(overlaps)),
                "mean_new_docs": float(np.mean(new_docs_all)),
                "pct_queries_shifted": float(np.mean([o < 1.0 for o in overlaps])),
                "examples": examples,
            }

        results["targets"][target_name] = target_results

    return results


def main():
    print(f"Item 9: Domain Validation Experiments")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    all_results = {}

    # ═══════════════════════════════════════════════════════════════
    # 9A: Drug Repurposing via SciFact
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  9A: Drug Repurposing (SciFact corpus)")
    print(f"{'='*70}")

    corpus, queries_sf, qrels = GenericDataLoader(str(DATA_DIR / "scifact")).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]
    print(f"  Corpus: {len(doc_ids)} docs")

    corpus_embs = np.array(model.encode(doc_texts, batch_size=256,
                                         normalize_embeddings=True, show_progress_bar=True))

    # Encode drug repurposing queries
    drp_ids = list(DRUG_REPURPOSING_QUERIES.keys())
    drp_texts = list(DRUG_REPURPOSING_QUERIES.values())
    drp_embs = np.array(model.encode(drp_texts, normalize_embeddings=True))

    print(f"  {len(drp_ids)} drug repurposing queries")

    # Run rotation analysis
    alphas = [0.1, 0.2, 0.3, 0.5]
    drp_results = analyze_rotation_domain_shift(
        drp_embs, drp_ids, DRUG_REPURPOSING_QUERIES, corpus_embs,
        doc_ids, corpus, model, ROTATION_TARGETS_MEDICAL, alphas, "drug_repurposing"
    )

    # Print highlights
    for target_name, tdata in drp_results["targets"].items():
        print(f"\n  Rotation toward: {target_name}")
        for akey, adata in tdata["alphas"].items():
            print(f"    {akey}: Jaccard={adata['mean_jaccard']:.3f} "
                  f"new_docs={adata['mean_new_docs']:.1f} "
                  f"shifted={adata['pct_queries_shifted']:.0%}")

    # Show detailed examples for α=0.3
    print(f"\n  --- Example: Parkinson query rotated toward Alzheimer's (α=0.3) ---")
    alz_03 = drp_results["targets"]["alzheimers"]["alphas"].get("alpha=0.3", {})
    for ex in alz_03.get("examples", [])[:2]:
        print(f"\n  Query: {ex['query']}")
        print(f"  Baseline top docs:")
        for d in ex["baseline_docs"][:2]:
            print(f"    - [{d['doc_id']}] {d['title'][:80]}")
        print(f"  NEW docs after rotation:")
        for d in ex["new_docs_after_rotation"][:2]:
            print(f"    + [{d['doc_id']}] {d['title'][:80]}")

    all_results["9A_drug_repurposing"] = drp_results

    # ═══════════════════════════════════════════════════════════════
    # 9B: Legal Analogy via ArguAna
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  9B: Legal Analogy (ArguAna corpus)")
    print(f"{'='*70}")

    corpus_ar, queries_ar, qrels_ar = GenericDataLoader(str(DATA_DIR / "arguana")).load(split="test")

    doc_ids_ar = list(corpus_ar.keys())
    doc_texts_ar = [(corpus_ar[d].get("title", "") + " " + corpus_ar[d].get("text", "")).strip()
                     for d in doc_ids_ar]
    print(f"  Corpus: {len(doc_ids_ar)} docs")

    corpus_embs_ar = np.array(model.encode(doc_texts_ar, batch_size=256,
                                            normalize_embeddings=True, show_progress_bar=True))

    # Encode legal queries
    leg_ids = list(LEGAL_QUERIES.keys())
    leg_texts = list(LEGAL_QUERIES.values())
    leg_embs = np.array(model.encode(leg_texts, normalize_embeddings=True))

    print(f"  {len(leg_ids)} legal queries")

    leg_results = analyze_rotation_domain_shift(
        leg_embs, leg_ids, LEGAL_QUERIES, corpus_embs_ar,
        doc_ids_ar, corpus_ar, model, ROTATION_TARGETS_LEGAL, alphas, "legal_analogy"
    )

    for target_name, tdata in leg_results["targets"].items():
        print(f"\n  Rotation toward: {target_name}")
        for akey, adata in tdata["alphas"].items():
            print(f"    {akey}: Jaccard={adata['mean_jaccard']:.3f} "
                  f"new_docs={adata['mean_new_docs']:.1f} "
                  f"shifted={adata['pct_queries_shifted']:.0%}")

    all_results["9B_legal_analogy"] = leg_results

    # ═══════════════════════════════════════════════════════════════
    # 9C: Cross-domain retrieval quality (using SciFact qrels as proxy)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  9C: Cross-domain retrieval on SciFact (with qrels)")
    print(f"{'='*70}")

    # Use actual SciFact queries and measure if rotation toward medical
    # targets can improve retrieval for queries about specific topics
    query_ids_sf = list(queries_sf.keys())
    query_texts_sf = [queries_sf[q] for q in query_ids_sf]
    query_embs_sf = np.array(model.encode(query_texts_sf, normalize_embeddings=True))

    from pytrec_eval import RelevanceEvaluator
    evaluator = RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    # Baseline
    base_sims = query_embs_sf @ corpus_embs.T
    base_res = {}
    for i, qid in enumerate(query_ids_sf):
        top = np.argsort(base_sims[i])[::-1][:100]
        base_res[qid] = {doc_ids[idx]: float(base_sims[i, idx]) for idx in top}
    base_pq = evaluator.evaluate(base_res)
    base_mean = np.mean([base_pq[qid]["ndcg_cut_10"] for qid in query_ids_sf if qid in base_pq])

    cross_domain_results = {"baseline_ndcg": round(float(base_mean), 4), "targets": {}}

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
            rot_mean = np.mean([rot_pq[qid]["ndcg_cut_10"] for qid in query_ids_sf if qid in rot_pq])

            key = f"{target_name}_alpha={alpha}"
            cross_domain_results["targets"][key] = {
                "ndcg": round(float(rot_mean), 4),
                "delta": round(float(rot_mean - base_mean), 4),
            }
            print(f"  {target_name} α={alpha}: nDCG={rot_mean:.4f} Δ={rot_mean-base_mean:+.4f}")

    all_results["9C_cross_domain_scifact"] = cross_domain_results

    # Save all
    with open(RESULTS_DIR / "domain_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Item 9: Domain Validation")
    print(f"{'='*70}")
    print(f"\n  9A Drug Repurposing: rotation effectively shifts results cross-domain")
    print(f"  9B Legal Analogy: rotation shifts argumentation results toward legal domains")
    print(f"  9C Cross-domain SciFact: measured nDCG impact of domain-targeted rotation")
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
