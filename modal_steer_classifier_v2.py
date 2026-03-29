"""
STEER — Classifier v2: Multi-Operation Best-of-N

MUDANÇAS vs v1:
1. LABEL: Em vez de testar apenas steer(q, t, 0.1), testa 7 operações e marca
   como positivo se QUALQUER uma melhora. Isso aumenta a taxa de positivos
   de 3-34% para (estimativa) 30-60%.

2. LABEL MULTI-CLASSE: Além de binário (beneficia/não), prediz QUAL operação
   é a melhor para cada query. Isso transforma o classifier de "gate" em "router".

3. FEATURES EXTRAS: Adiciona features derivadas dos resultados da Wave 1:
   - query_neg_affinity: cos_sim com o target negativo (prediz se rotate_away ajuda)
   - centroid_alignment: cos_sim(q, centroid) (prediz amplify vs diffuse)
   - target_orthogonality: 1 - |cos_sim(q, t_pos) - cos_sim(q, t_neg)| (prediz contrastive)

O objetivo: um router que recebe uma query e diz "use contrastive com α=0.1/β=0.2"
ou "use rotate_away com α=0.2" ou "não steere".

6 models x 5 datasets. 5-fold CV + cross-dataset transfer.

Usage: modal run modal_steer_classifier_v2.py
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

app = modal.App("steer-classifier-v2", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("all-mpnet-base-v2", "trained"),
    ("BAAI/bge-base-en-v1.5", "contrastive"),
    ("intfloat/e5-small-v2", "instruction"),
    ("thenlper/gte-small", "general"),
]

DATASETS_CONFIG = {
    "scifact": {
        "target_positive": "clinical medicine and patient outcomes",
        "target_negative": "animal model laboratory studies",
    },
    "arguana": {
        "target_positive": "legal reasoning and jurisprudence",
        "target_negative": "informal opinion and anecdote",
    },
    "nfcorpus": {
        "target_positive": "clinical nutrition interventions",
        "target_negative": "food industry marketing",
    },
    "fiqa": {
        "target_positive": "macroeconomic policy impacts",
        "target_negative": "personal finance anecdotes",
    },
    "trec-covid": {
        "target_positive": "COVID-19 clinical treatment protocols",
        "target_negative": "COVID-19 misinformation and conspiracy",
    },
}

MAX_QUERIES = 300
IMPROVEMENT_THRESHOLD = 0.001  # Min nDCG improvement to count as positive


@app.function(gpu="T4", memory=16384, timeout=5400, volumes={"/results": vol})
def run_classifier_v2(model_name: str, family: str):
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    print(f"\n{'='*60}")
    print(f"  Classifier v2: {model_name} ({family})")
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

    def rrf_fusion_single(scores_list, k=60):
        n = len(scores_list[0])
        rrf = np.zeros(n)
        for s in scores_list:
            ranks = np.argsort(np.argsort(-s)) + 1
            rrf += 1.0 / (k + ranks)
        return rrf

    def ndcg_from_scores(scores, doc_ids, qrels_qid, k=10):
        top_idx = np.argsort(scores)[::-1][:k]
        dcg = sum(qrels_qid.get(doc_ids[idx], 0) / np.log2(r + 2) for r, idx in enumerate(top_idx))
        rels = sorted(qrels_qid.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(j + 2) for j, rel in enumerate(rels))
        return dcg / max(idcg, 1e-10)

    model_results = {"model": model_name, "family": family}
    all_features = {}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
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
        pos_emb = embed_model.encode(ds_cfg["target_positive"], normalize_embeddings=True)
        neg_emb = embed_model.encode(ds_cfg["target_negative"], normalize_embeddings=True)

        centroid = normalize_vec(np.mean(corpus_embs, axis=0))

        # IDF
        all_terms = Counter()
        for text in doc_texts:
            all_terms.update(text.lower().split())
        N = len(doc_texts)
        idf = {t: np.log(N / (1 + c)) for t, c in all_terms.items()}

        # === COMPUTE 7 OPERATIONS PER QUERY ===
        OPERATIONS = [
            "rotate_toward_0.1",
            "rotate_toward_0.2",
            "rotate_away_0.1",
            "rotate_away_0.2",
            "contrastive_sym_0.1",
            "contrastive_asym_a0.1_b0.2",
            "amplify_0.1",
        ]

        features_list = []
        labels_binary = []  # 1 if ANY operation helps
        labels_best_op = []  # index of best operation (multi-class)
        labels_best_name = []  # name of best operation

        for i, (qid, qt) in enumerate(zip(query_ids, query_texts)):
            q = query_embs[i]
            base_scores = q @ corpus_embs.T
            ndcg_base = ndcg_from_scores(base_scores, doc_ids, qrels[qid])

            # Test each operation
            op_deltas = {}

            # Rotate toward
            for alpha in [0.1, 0.2]:
                q_rt = normalize_vec(q + alpha * pos_emb)
                ndcg_rt = ndcg_from_scores(q_rt @ corpus_embs.T, doc_ids, qrels[qid])
                op_deltas[f"rotate_toward_{alpha}"] = ndcg_rt - ndcg_base

            # Rotate away
            for alpha in [0.1, 0.2]:
                q_ra = normalize_vec(q - alpha * pos_emb)
                ndcg_ra = ndcg_from_scores(q_ra @ corpus_embs.T, doc_ids, qrels[qid])
                op_deltas[f"rotate_away_{alpha}"] = ndcg_ra - ndcg_base

            # Contrastive symmetric
            q_cs = normalize_vec(q + 0.1 * pos_emb - 0.1 * neg_emb)
            ndcg_cs = ndcg_from_scores(q_cs @ corpus_embs.T, doc_ids, qrels[qid])
            op_deltas["contrastive_sym_0.1"] = ndcg_cs - ndcg_base

            # Contrastive asymmetric (best config from Wave 2)
            q_ca = normalize_vec(q + 0.1 * pos_emb - 0.2 * neg_emb)
            ndcg_ca = ndcg_from_scores(q_ca @ corpus_embs.T, doc_ids, qrels[qid])
            op_deltas["contrastive_asym_a0.1_b0.2"] = ndcg_ca - ndcg_base

            # Amplify
            q_amp = normalize_vec(q + 0.1 * centroid)
            ndcg_amp = ndcg_from_scores(q_amp @ corpus_embs.T, doc_ids, qrels[qid])
            op_deltas["amplify_0.1"] = ndcg_amp - ndcg_base

            # Labels
            best_op = max(op_deltas, key=op_deltas.get)
            best_delta = op_deltas[best_op]
            any_positive = any(d > IMPROVEMENT_THRESHOLD for d in op_deltas.values())

            labels_binary.append(1 if any_positive else 0)
            labels_best_op.append(OPERATIONS.index(best_op) if best_delta > IMPROVEMENT_THRESHOLD else len(OPERATIONS))  # last class = "no steer"
            labels_best_name.append(best_op if best_delta > IMPROVEMENT_THRESHOLD else "none")

            # Features (9 total)
            query_length = len(qt.split())
            terms = qt.lower().split()
            specificity = float(np.mean([idf.get(t, np.log(N)) for t in terms])) if terms else 0
            dist_to_centroid = 1.0 - float(q @ centroid)
            embedding_norm = float(np.linalg.norm(q))

            sims_to_corpus = q @ corpus_embs.T
            top10_sims = np.sort(sims_to_corpus)[::-1][:10]
            isotropy_local = float(np.mean(top10_sims))

            query_pos_sim = float(q @ pos_emb)
            query_neg_sim = float(q @ neg_emb)
            centroid_alignment = float(q @ centroid)
            target_orthogonality = 1.0 - abs(query_pos_sim - query_neg_sim)

            features_list.append([
                query_length, specificity, dist_to_centroid,
                embedding_norm, isotropy_local, query_pos_sim,
                query_neg_sim, centroid_alignment, target_orthogonality,
            ])

        X = np.array(features_list)
        y_bin = np.array(labels_binary)
        y_multi = np.array(labels_best_op)

        all_features[ds_name] = (X, y_bin, y_multi, labels_best_name)

        pos_rate = float(np.mean(y_bin))
        print(f"    Positive rate (any op helps): {pos_rate:.1%}")
        print(f"    Best op distribution: {Counter(labels_best_name).most_common()}")

        if len(np.unique(y_bin)) < 2:
            print(f"    WARNING: Only one class. Skipping.")
            model_results[ds_name] = {"error": "single_class", "positive_rate": pos_rate}
            continue

        # === BINARY CLASSIFIER: should we steer at all? ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        feature_names = [
            "query_length", "specificity", "dist_to_centroid",
            "embedding_norm", "isotropy_local", "query_pos_sim",
            "query_neg_sim", "centroid_alignment", "target_orthogonality",
        ]

        # Test 3 classifiers
        classifiers = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        binary_results = {}

        for clf_name, clf in classifiers.items():
            try:
                cv_acc = cross_val_score(clf, X_scaled, y_bin, cv=cv, scoring="accuracy")
                cv_f1 = cross_val_score(clf, X_scaled, y_bin, cv=cv, scoring="f1")
                cv_prec = cross_val_score(clf, X_scaled, y_bin, cv=cv, scoring="precision")
                cv_rec = cross_val_score(clf, X_scaled, y_bin, cv=cv, scoring="recall")

                binary_results[clf_name] = {
                    "accuracy": round(float(np.mean(cv_acc)), 4),
                    "f1": round(float(np.mean(cv_f1)), 4),
                    "precision": round(float(np.mean(cv_prec)), 4),
                    "recall": round(float(np.mean(cv_rec)), 4),
                }
                print(f"    Binary {clf_name}: Acc={np.mean(cv_acc):.3f} F1={np.mean(cv_f1):.3f} P={np.mean(cv_prec):.3f} R={np.mean(cv_rec):.3f}")
            except Exception as e:
                binary_results[clf_name] = {"error": str(e)}

        # Feature importances (from best binary classifier)
        best_clf_name = max(binary_results, key=lambda k: binary_results[k].get("f1", 0))
        best_clf = classifiers[best_clf_name]
        best_clf.fit(X_scaled, y_bin)
        if hasattr(best_clf, 'feature_importances_'):
            importances = dict(zip(feature_names, [round(float(c), 4) for c in best_clf.feature_importances_]))
        elif hasattr(best_clf, 'coef_'):
            importances = dict(zip(feature_names, [round(float(abs(c)), 4) for c in best_clf.coef_[0]]))
        else:
            importances = {}

        # === MULTI-CLASS CLASSIFIER: which operation is best? ===
        # Only if enough classes have samples
        n_classes_with_samples = len([c for c in np.unique(y_multi) if np.sum(y_multi == c) >= 5])
        multi_results = {}

        if n_classes_with_samples >= 3:
            clf_multi = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            try:
                cv_acc_m = cross_val_score(clf_multi, X_scaled, y_multi, cv=cv, scoring="accuracy")
                cv_f1_m = cross_val_score(clf_multi, X_scaled, y_multi, cv=cv, scoring="f1_macro")
                multi_results = {
                    "accuracy": round(float(np.mean(cv_acc_m)), 4),
                    "f1_macro": round(float(np.mean(cv_f1_m)), 4),
                    "n_classes": int(n_classes_with_samples),
                }
                print(f"    Multi-class RF: Acc={np.mean(cv_acc_m):.3f} F1_macro={np.mean(cv_f1_m):.3f} ({n_classes_with_samples} classes)")
            except Exception as e:
                multi_results = {"error": str(e)}
        else:
            print(f"    Multi-class: skipped (only {n_classes_with_samples} classes with >=5 samples)")

        ds_result = {
            "n_queries": len(query_ids),
            "positive_rate_any_op": round(pos_rate, 4),
            "best_op_distribution": dict(Counter(labels_best_name).most_common()),
            "binary_classifiers": binary_results,
            "best_binary_classifier": best_clf_name,
            "feature_importances": importances,
            "top_feature": max(importances, key=importances.get) if importances else "unknown",
            "multi_class": multi_results,
        }

        model_results[ds_name] = ds_result

    # Cross-dataset transfer: train on trec-covid (highest pos rate), test on fiqa
    if "trec-covid" in all_features and "fiqa" in all_features:
        X_train, y_train, _, _ = all_features["trec-covid"]
        X_test, y_test, _, _ = all_features["fiqa"]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            model_results["cross_dataset_transfer"] = {
                "train": "trec-covid",
                "test": "fiqa",
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            }
            print(f"\n  Transfer trec-covid→fiqa: Acc={accuracy_score(y_test, y_pred):.3f} F1={f1_score(y_test, y_pred, zero_division=0):.3f}")

    safe = model_name.replace("/", "_")
    out_path = f"/results/steer_classifier_v2/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Classifier v2: Multi-Op Router (6 models x 5 datasets)")
    print("=" * 70)
    results = list(run_classifier_v2.starmap([(n, f) for n, f in MODELS]))
    print("\n  DONE!")
