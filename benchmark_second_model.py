"""
Item 6: Run BEIR benchmark with MiniLM (second model) on SciFact + ArguAna.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from a2rag import rotate_toward, subtract_orthogonal

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data/beir")
RESULTS_DIR = Path("results/deep_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["scifact", "arguana"]
ROTATION_TARGETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
}
SUBTRACTION = {
    "scifact": "methodology and statistical analysis",
    "arguana": "economic arguments",
}

print(f"Model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
all_results = {}

for ds in DATASETS:
    print(f"\n{'='*50}\n  {ds}\n{'='*50}")
    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR / ds)).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title","")+" "+corpus[d].get("text","")).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    print(f"  Encoding {len(doc_texts)} docs...")
    corpus_embs = np.array(model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
    query_embs = np.array(model.encode(query_texts, normalize_embeddings=True))

    evaluator = EvaluateRetrieval()
    r = {}

    # Baseline
    sims = query_embs @ corpus_embs.T
    base = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(sims[i])[::-1][:100]
        base[qid] = {doc_ids[idx]: float(sims[i,idx]) for idx in top}
    ndcg, m, rec, p = evaluator.evaluate(qrels, base, [10])
    r["baseline"] = ndcg.get("NDCG@10", 0)
    print(f"  Baseline: nDCG@10={r['baseline']:.4f}")

    # Rotation
    target_emb = model.encode(ROTATION_TARGETS[ds], normalize_embeddings=True)
    for alpha in [0.1, 0.2, 0.3]:
        rot = np.array([rotate_toward(q, target_emb, alpha) for q in query_embs])
        rot_sims = rot @ corpus_embs.T
        rot_res = {}
        for i, qid in enumerate(query_ids):
            top = np.argsort(rot_sims[i])[::-1][:100]
            rot_res[qid] = {doc_ids[idx]: float(rot_sims[i,idx]) for idx in top}
        ndcg, _, _, _ = evaluator.evaluate(qrels, rot_res, [10])
        r[f"rot_a{alpha}"] = ndcg.get("NDCG@10", 0)
        print(f"  Rot α={alpha}: nDCG@10={r[f'rot_a{alpha}']:.4f} Δ={r[f'rot_a{alpha}']-r['baseline']:+.4f}")

    # Subtraction
    concept_emb = model.encode(SUBTRACTION[ds], normalize_embeddings=True)
    sub = np.array([subtract_orthogonal(q, concept_emb) for q in query_embs])
    sub_sims = sub @ corpus_embs.T
    sub_res = {}
    for i, qid in enumerate(query_ids):
        top = np.argsort(sub_sims[i])[::-1][:100]
        sub_res[qid] = {doc_ids[idx]: float(sub_sims[i,idx]) for idx in top}
    ndcg, _, _, _ = evaluator.evaluate(qrels, sub_res, [10])
    r["subtraction"] = ndcg.get("NDCG@10", 0)
    print(f"  Sub: nDCG@10={r['subtraction']:.4f} Δ={r['subtraction']-r['baseline']:+.4f}")

    all_results[ds] = r

with open(RESULTS_DIR / "second_model_minilm.json", "w") as f:
    json.dump({"model": MODEL_NAME, "results": all_results}, f, indent=2)

print("\n\nSUMMARY (MiniLM-L6-v2):")
print(f"{'Dataset':<12} {'Baseline':<10} {'Rot 0.1':<10} {'Rot 0.2':<10} {'Rot 0.3':<10} {'Sub':<10}")
for ds, r in all_results.items():
    print(f"{ds:<12} {r['baseline']:<10.4f} {r['rot_a0.1']:<10.4f} {r['rot_a0.2']:<10.4f} {r['rot_a0.3']:<10.4f} {r['subtraction']:<10.4f}")
