"""
STEER — Steer Classifier (Meta-Learning)

Trains logistic regression to predict which queries benefit from steering.
Features: query_length, specificity (IDF), dist_to_centroid, embedding_norm,
isotropy_local, query_target_sim.
Label: 1 if steer improves per-query nDCG@10, 0 otherwise.

5-fold CV + cross-dataset transfer (train scifact -> test fiqa).

6 models x 5 datasets.

Usage: modal run modal_steer_classifier.py
Author: Renato Aparecido Gomes
"""

import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0", "beir", "torch", "numpy",
        "scipy", "pytrec_eval", "datasets", "faiss-cpu", "scikit-learn",
    )
)

app = modal.App("steer-classifier", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("all-mpnet-base-v2", "trained"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("intfloat/e5-small-v2", "instruction"),
    ("thenlper/gte-small", "general"),
]

DATASETS = {
    "scifact": "clinical medicine and patient outcomes",
    "arguana": "legal reasoning and jurisprudence",
    "nfcorpus": "clinical nutrition interventions",
    "fiqa": "macroeconomic policy impacts",
    "trec-covid": "COVID-19 clinical treatment protocols",
}

ALPHA = 0.1
MAX_QUERIES = 300  # More queries for better classifier training


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_classifier(model_name: str, family: str):
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Steer Classifier: {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def per_query_ndcg_single(query_emb, corpus_embs, doc_ids, qrels_qid, k=10):
        scores = query_emb @ corpus_embs.T
        top_idx = np.argsort(scores)[::-1][:k]
        dcg = sum(qrels_qid.get(doc_ids[idx], 0) / np.log2(r + 2) for r, idx in enumerate(top_idx))
        rels = sorted(qrels_qid.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(j + 2) for j, rel in enumerate(rels))
        return dcg / max(idcg, 1e-10)

    model_results = {"model": model_name, "family": family}
    all_features = {}  # ds_name -> (X, y)

    for ds_name, target_text in DATASETS.items():
        print(f"\n  Dataset: {ds_name}")
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, f"/tmp/beir-{ds_name}")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        except Exception as e:
            model_results[ds_name] = {"error": str(e)}
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
        query_ids = [qid for qid in list(queries.keys())[:MAX_QUERIES] if qid in qrels]
        query_texts = [queries[q] for q in query_ids]
        print(f"    Queries: {len(query_ids)}, Docs: {len(doc_ids)}")

        corpus_embs = np.array(embed_model.encode(doc_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True))
        query_embs = np.array(embed_model.encode(query_texts, normalize_embeddings=True))
        target_emb = embed_model.encode(target_text, normalize_embeddings=True)

        # Corpus centroid
        centroid = normalize_vec(np.mean(corpus_embs, axis=0))

        # IDF approximation
        all_terms = Counter()
        for text in doc_texts:
            all_terms.update(text.lower().split())
        N = len(doc_texts)
        idf = {t: np.log(N / (1 + c)) for t, c in all_terms.items()}

        # Compute features and labels
        features = []
        labels = []
        for i, (qid, qt) in enumerate(zip(query_ids, query_texts)):
            # Baseline nDCG
            ndcg_base = per_query_ndcg_single(query_embs[i], corpus_embs, doc_ids, qrels[qid])

            # Steered nDCG
            q_steered = normalize_vec(query_embs[i] + ALPHA * target_emb)
            ndcg_steered = per_query_ndcg_single(q_steered, corpus_embs, doc_ids, qrels[qid])

            # Label: 1 if steered improves
            label = 1 if ndcg_steered > ndcg_base + 0.001 else 0

            # Features
            query_length = len(qt.split())
            terms = qt.lower().split()
            specificity = float(np.mean([idf.get(t, np.log(N)) for t in terms])) if terms else 0
            dist_to_centroid = 1.0 - float(query_embs[i] @ centroid)
            embedding_norm = float(np.linalg.norm(query_embs[i]))

            # Local isotropy: mean cos_sim to 10 nearest neighbors
            sims_to_corpus = query_embs[i] @ corpus_embs.T
            top10_sims = np.sort(sims_to_corpus)[::-1][:10]
            isotropy_local = float(np.mean(top10_sims))

            query_target_sim = float(query_embs[i] @ target_emb)

            features.append([query_length, specificity, dist_to_centroid,
                           embedding_norm, isotropy_local, query_target_sim])
            labels.append(label)

        X = np.array(features)
        y = np.array(labels)
        all_features[ds_name] = (X, y)

        # 5-fold CV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=42)

        if len(np.unique(y)) < 2:
            print(f"    WARNING: Only one class present. Skipping.")
            model_results[ds_name] = {"error": "single_class", "positive_rate": float(np.mean(y))}
            continue

        cv_acc = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
        cv_f1 = cross_val_score(clf, X_scaled, y, cv=5, scoring="f1")
        cv_prec = cross_val_score(clf, X_scaled, y, cv=5, scoring="precision")
        cv_rec = cross_val_score(clf, X_scaled, y, cv=5, scoring="recall")

        # Feature importances (train on all data)
        clf.fit(X_scaled, y)
        feature_names = ["query_length", "specificity", "dist_to_centroid",
                        "embedding_norm", "isotropy_local", "query_target_sim"]
        importances = dict(zip(feature_names, [round(float(c), 4) for c in clf.coef_[0]]))

        ds_result = {
            "n_queries": len(query_ids),
            "positive_rate": round(float(np.mean(y)), 4),
            "cv_accuracy": round(float(np.mean(cv_acc)), 4),
            "cv_f1": round(float(np.mean(cv_f1)), 4),
            "cv_precision": round(float(np.mean(cv_prec)), 4),
            "cv_recall": round(float(np.mean(cv_rec)), 4),
            "feature_importances": importances,
            "top_feature": max(importances, key=lambda k: abs(importances[k])),
        }
        print(f"    Acc={np.mean(cv_acc):.3f} F1={np.mean(cv_f1):.3f} Top={ds_result['top_feature']}")
        model_results[ds_name] = ds_result

    # Cross-dataset transfer: train on scifact, test on fiqa
    if "scifact" in all_features and "fiqa" in all_features:
        X_train, y_train = all_features["scifact"]
        X_test, y_test = all_features["fiqa"]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_tr, y_train)
            from sklearn.metrics import accuracy_score, f1_score
            y_pred = clf.predict(X_te)
            model_results["cross_dataset_transfer"] = {
                "train": "scifact",
                "test": "fiqa",
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            }
            print(f"\n  Transfer scifact→fiqa: Acc={accuracy_score(y_test, y_pred):.3f} F1={f1_score(y_test, y_pred, zero_division=0):.3f}")

    safe = model_name.replace("/", "_")
    out_path = f"/results/steer_classifier/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Classifier (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_classifier.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
