"""
A2RAG — Cross-Domain Evaluation with LLM-Annotated Qrels

BEIR nao mede cross-domain retrieval. Esta avaliacao cria um setup onde
cross-domain E premiado, mostrando o valor real do A2RAG.

Design:
1. Usar 2 pares cross-domain naturais:
   - SciFact queries → NFCorpus docs (biomedical → clinical nutrition)
   - SciFact queries → TREC-COVID docs (biomedical → COVID clinical)

2. Para cada query, buscar top-20 docs no corpus cross-domain usando:
   - Baseline (sem rotation)
   - Addition alpha=0.1 (rotation leve)
   - Addition alpha=0.2 (rotation media)

3. LLM (Qwen2.5-7B) avalia cada par (query, doc) com score 0-2:
   0 = irrelevante
   1 = parcialmente relevante (conceito relacionado)
   2 = altamente relevante (responderia a query de outro angulo)

4. Computar cross-domain nDCG@10 usando os qrels do LLM.
   Se A2RAG surfaca mais docs relevantes cross-domain, nDCG sobe.

Usage: modal run modal_crossdomain_eval.py

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

app = modal.App("a2rag-crossdomain-eval", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

# Use 3 representative models (low/mid/high isotropy)
MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),      # low isotropy, responds well
    ("BAAI/bge-base-en-v1.5", "contrastive"), # high isotropy, degrades with rotation
    ("thenlper/gte-small", "general"),         # high isotropy
]

# Cross-domain pairs: source queries → target corpus
CROSS_DOMAIN_PAIRS = [
    {
        "name": "scifact_to_nfcorpus",
        "source_ds": "scifact",
        "target_ds": "nfcorpus",
        "rotation_target": "clinical nutrition and dietary interventions",
        "description": "Biomedical research queries searching clinical nutrition corpus",
    },
    {
        "name": "scifact_to_trec-covid",
        "source_ds": "scifact",
        "target_ds": "trec-covid",
        "rotation_target": "COVID-19 clinical management and treatment",
        "description": "Biomedical research queries searching COVID clinical corpus",
    },
]

ALPHAS = [0.0, 0.1, 0.2, 0.3]
MAX_QUERIES = 50  # Limit for LLM annotation cost
TOP_K_ANNOTATE = 20  # Top docs to annotate per query per strategy


@app.function(gpu="L4", memory=32768, timeout=5400, volumes={"/results": vol})
def run_crossdomain_eval(model_name: str, family: str):
    """Run cross-domain evaluation for one model."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc

    print(f"\n{'='*60}")
    print(f"  Cross-Domain Eval: {model_name} ({family})")
    print(f"{'='*60}")

    # Load embedding model
    embed_model = SentenceTransformer(model_name)

    # Load LLM for annotation
    print("  Loading Qwen2.5-7B-Instruct for annotation...")
    llm_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float16, device_map="auto"
    )

    def annotate_relevance(query: str, doc: str) -> int:
        """LLM judges cross-domain relevance: 0, 1, or 2."""
        prompt = f"""You are an expert scientific reviewer. Judge if the following document is relevant to the query, considering CROSS-DOMAIN relevance (the document may be from a different field but still informative).

Query: {query[:500]}

Document: {doc[:800]}

Score:
0 = Not relevant at all (completely different topic)
1 = Partially relevant (related concept, could provide useful context)
2 = Highly relevant (directly addresses the query from a different angle/domain)

Reply with ONLY the number (0, 1, or 2):"""

        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm.generate(
                **inputs, max_new_tokens=5, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Parse score
        for char in response:
            if char in "012":
                return int(char)
        return 0  # Default to irrelevant if parsing fails

    model_results = {"model": model_name, "family": family}

    for pair in CROSS_DOMAIN_PAIRS:
        print(f"\n  Pair: {pair['name']} — {pair['description']}")

        # Load source queries
        src_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{pair['source_ds']}.zip"
        src_path = util.download_and_unzip(src_url, f"/tmp/beir-{pair['source_ds']}")
        src_corpus, src_queries, src_qrels = GenericDataLoader(src_path).load(split="test")

        # Load target corpus
        tgt_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{pair['target_ds']}.zip"
        tgt_path = util.download_and_unzip(tgt_url, f"/tmp/beir-{pair['target_ds']}")
        tgt_corpus, _, _ = GenericDataLoader(tgt_path).load(split="test")

        # Prepare data
        tgt_doc_ids = list(tgt_corpus.keys())
        tgt_doc_texts = [(tgt_corpus[d].get("title", "") + " " + tgt_corpus[d].get("text", "")).strip() for d in tgt_doc_ids]

        # Select queries (prioritize those with more relevance judgments in source)
        query_ids = list(src_queries.keys())[:MAX_QUERIES]
        query_texts = [src_queries[q] for q in query_ids]

        print(f"    Queries: {len(query_ids)}, Target docs: {len(tgt_doc_ids)}")

        # Encode
        tgt_embs = np.array(embed_model.encode(tgt_doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False))
        target_emb = embed_model.encode(pair["rotation_target"], normalize_embeddings=True)

        # For each alpha, get top-K docs and collect unique (query, doc) pairs for annotation
        pairs_to_annotate = {}  # (qid, doc_id) -> (query_text, doc_text)

        alpha_results_map = {}  # alpha -> {qid -> [(doc_id, sim_score), ...]}

        for alpha in ALPHAS:
            if alpha == 0:
                q_embs = query_embs
            else:
                q_embs = query_embs + alpha * target_emb
                norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
                q_embs = q_embs / np.maximum(norms, 1e-10)

            sims = q_embs @ tgt_embs.T
            alpha_docs = {}

            for i, qid in enumerate(query_ids):
                top = np.argsort(sims[i])[::-1][:TOP_K_ANNOTATE]
                alpha_docs[qid] = [(tgt_doc_ids[idx], float(sims[i, idx])) for idx in top]

                for idx in top:
                    key = (qid, tgt_doc_ids[idx])
                    if key not in pairs_to_annotate:
                        pairs_to_annotate[key] = (src_queries[qid], tgt_doc_texts[idx])

            alpha_results_map[alpha] = alpha_docs

        print(f"    Unique pairs to annotate: {len(pairs_to_annotate)}")

        # Annotate all unique pairs with LLM
        annotations = {}  # (qid, doc_id) -> score
        annotated = 0
        for (qid, doc_id), (q_text, d_text) in pairs_to_annotate.items():
            score = annotate_relevance(q_text, d_text)
            annotations[(qid, doc_id)] = score
            annotated += 1
            if annotated % 100 == 0:
                print(f"    Annotated {annotated}/{len(pairs_to_annotate)} pairs...")

        print(f"    Annotation complete: {annotated} pairs")

        # Score distribution
        score_counts = {0: 0, 1: 0, 2: 0}
        for s in annotations.values():
            score_counts[s] = score_counts.get(s, 0) + 1
        print(f"    Score distribution: 0={score_counts[0]}, 1={score_counts[1]}, 2={score_counts[2]}")

        # Build cross-domain qrels from annotations
        cross_qrels = {}
        for (qid, doc_id), score in annotations.items():
            if score > 0:  # Only include relevant docs
                cross_qrels.setdefault(qid, {})[doc_id] = score

        # Evaluate each alpha strategy using cross-domain qrels
        pair_results = {
            "n_queries": len(query_ids),
            "n_target_docs": len(tgt_doc_ids),
            "n_annotated_pairs": len(annotations),
            "score_distribution": score_counts,
            "n_queries_with_relevant": len(cross_qrels),
        }

        if len(cross_qrels) < 5:
            print(f"    WARNING: Only {len(cross_qrels)} queries have relevant docs. Results may be noisy.")

        for alpha in ALPHAS:
            # Build retrieval results for this alpha
            results_for_eval = {}
            for qid in query_ids:
                if qid not in cross_qrels:
                    continue
                doc_scores = {doc_id: sim for doc_id, sim in alpha_results_map[alpha][qid]}
                results_for_eval[qid] = doc_scores

            if not results_for_eval:
                pair_results["baseline" if alpha == 0 else f"alpha_{alpha}"] = {"error": "no queries with relevant docs"}
                continue

            # Manual nDCG computation (avoid pytrec_eval import issues)
            def compute_ndcg_at_k(qrels_dict, results_dict, k=10):
                """Compute nDCG@k manually."""
                ndcgs = []
                for qid in results_dict:
                    if qid not in qrels_dict:
                        continue
                    # Sort docs by score
                    sorted_docs = sorted(results_dict[qid].items(), key=lambda x: x[1], reverse=True)[:k]
                    # DCG
                    dcg = 0
                    for rank, (doc_id, _) in enumerate(sorted_docs):
                        rel = qrels_dict.get(qid, {}).get(doc_id, 0)
                        dcg += rel / np.log2(rank + 2)
                    # Ideal DCG
                    ideal_rels = sorted(qrels_dict.get(qid, {}).values(), reverse=True)[:k]
                    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
                    ndcgs.append(dcg / idcg if idcg > 0 else 0)
                return float(np.mean(ndcgs)) if ndcgs else 0

            def compute_recall_at_k(qrels_dict, results_dict, k=20):
                """Compute Recall@k."""
                recalls = []
                for qid in results_dict:
                    if qid not in qrels_dict:
                        continue
                    sorted_docs = sorted(results_dict[qid].items(), key=lambda x: x[1], reverse=True)[:k]
                    retrieved_ids = {doc_id for doc_id, _ in sorted_docs}
                    relevant_ids = {doc_id for doc_id, rel in qrels_dict[qid].items() if rel > 0}
                    if relevant_ids:
                        recalls.append(len(retrieved_ids & relevant_ids) / len(relevant_ids))
                return float(np.mean(recalls)) if recalls else 0

            avg_metrics = {
                "ndcg_cut_10": round(compute_ndcg_at_k(cross_qrels, results_for_eval, 10), 4),
                "ndcg_cut_5": round(compute_ndcg_at_k(cross_qrels, results_for_eval, 5), 4),
                "recall_10": round(compute_recall_at_k(cross_qrels, results_for_eval, 10), 4),
                "recall_20": round(compute_recall_at_k(cross_qrels, results_for_eval, 20), 4),
            }

            # Count unique relevant docs surfaced
            relevant_surfaced = set()
            for qid in results_for_eval:
                for doc_id, _ in alpha_results_map[alpha][qid]:
                    if annotations.get((qid, doc_id), 0) > 0:
                        relevant_surfaced.add((qid, doc_id))

            label = "baseline" if alpha == 0 else f"alpha_{alpha}"
            pair_results[label] = {
                "metrics": avg_metrics,
                "relevant_surfaced": len(relevant_surfaced),
            }

            delta_str = ""
            if alpha > 0 and "baseline" in pair_results:
                base_ndcg = pair_results["baseline"]["metrics"].get("ndcg_cut_10", 0)
                this_ndcg = avg_metrics.get("ndcg_cut_10", 0)
                delta_str = f" (Δ={this_ndcg - base_ndcg:+.4f})"

            print(f"    α={alpha}: nDCG@10={avg_metrics.get('ndcg_cut_10', 0):.4f}{delta_str}"
                  f"  recall@20={avg_metrics.get('recall_20', 0):.4f}"
                  f"  relevant_surfaced={len(relevant_surfaced)}")

        model_results[pair["name"]] = pair_results

    # Save
    safe_name = model_name.replace("/", "_")
    out_path = f"/results/crossdomain_eval/{safe_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")

    # Cleanup
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  A2RAG — Cross-Domain Evaluation with LLM-Annotated Qrels")
    print("  3 models x 2 cross-domain pairs x 4 alphas")
    print("=" * 70)

    all_results = list(run_crossdomain_eval.starmap(
        [(name, fam) for name, fam in MODELS]
    ))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Cross-Domain nDCG@10")
    print("=" * 70)

    for r in all_results:
        print(f"\n  {r['model']} ({r['family']}):")
        for pair_name in ["scifact_to_nfcorpus", "scifact_to_trec-covid"]:
            if pair_name not in r:
                continue
            p = r[pair_name]
            base = p.get("baseline", {}).get("metrics", {}).get("ndcg_cut_10", 0)
            print(f"    {pair_name}: baseline={base:.4f}", end="")
            for alpha in [0.1, 0.2, 0.3]:
                a = p.get(f"alpha_{alpha}", {}).get("metrics", {}).get("ndcg_cut_10", 0)
                if a:
                    print(f"  α={alpha}={a:.4f}(Δ={a-base:+.4f})", end="")
            rel_base = p.get("baseline", {}).get("relevant_surfaced", 0)
            rel_03 = p.get("alpha_0.3", {}).get("relevant_surfaced", 0)
            print(f"  rel_docs: {rel_base}→{rel_03}")

    print("\n  DONE!")
