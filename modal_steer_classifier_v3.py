"""
STEER — Classifier v3: Per-Query LLM Targets + Multi-Op Router

MUDANÇA vs v2: Em vez de targets genéricos fixos por dataset, usa Qwen2.5-3B
para gerar targets específicos por query. Hipótese: per-query targets aumentam
a taxa de positivos em datasets onde targets genéricos falham (scifact, arguana).

7 operações testadas por query, cada uma com targets per-query:
1. Rotate Toward (α=0.1, α=0.2)
2. Rotate Away (α=0.1, α=0.2)
3. Contrastive asym (α+=0.1, β-=0.2) — com target negativo per-query
4. Multi-vector RRF (q + T(q), fusion)
5. Amplify (α=0.1, toward centroid)

3 classifiers: logistic, random_forest, gradient_boosting (all balanced).
5-fold stratified CV + cross-dataset transfer.

Requer L4 GPU (Qwen2.5-3B para target generation).

Usage: modal run modal_steer_classifier_v3.py
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
        "transformers>=4.40", "accelerate",
    )
)

app = modal.App("steer-classifier-v3", image=image)
vol = modal.Volume.from_name("a2rag-results", create_if_missing=True)

MODELS = [
    ("all-MiniLM-L6-v2", "distilled"),
    ("BAAI/bge-small-en-v1.5", "contrastive"),
    ("thenlper/gte-small", "general"),
]

DATASETS_CONFIG = {
    "scifact": {"domain": "biomedical research", "neg_generic": "animal model laboratory studies"},
    "arguana": {"domain": "argumentation and debate", "neg_generic": "informal opinion and anecdote"},
    "nfcorpus": {"domain": "nutrition and health", "neg_generic": "food industry marketing"},
    "fiqa": {"domain": "financial QA", "neg_generic": "personal finance anecdotes"},
    "trec-covid": {"domain": "COVID-19 research", "neg_generic": "COVID-19 misinformation and conspiracy"},
}

MAX_QUERIES = 200
IMPROVEMENT_THRESHOLD = 0.001
RRF_K = 60


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={"/results": vol})
def run_classifier_v3(model_name: str, family: str):
    import gc
    import numpy as np
    import torch
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  Classifier v3 (per-query targets): {model_name} ({family})")
    print(f"{'='*60}")

    embed_model = SentenceTransformer(model_name)

    # Load LLM for per-query targets
    print("  Loading Qwen2.5-3B-Instruct...")
    llm_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="auto")

    def generate_pos_neg_targets(query, domain):
        """Generate both positive (adjacent domain) and negative (noise) targets."""
        prompt = f"""Given a query from {domain}, suggest:
Line 1: One SHORT phrase (3-6 words) of an adjacent domain with relevant insights
Line 2: One SHORT phrase (3-6 words) of a domain that would add NOISE or be irrelevant

Query: {query[:300]}
Return EXACTLY 2 lines:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=60, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        lines = [l.strip().strip("-•·0123456789.):").strip() for l in resp.split("\n") if l.strip() and len(l.strip()) > 3]
        pos = lines[0] if len(lines) >= 1 else "related research findings"
        neg = lines[1] if len(lines) >= 2 else "unrelated general knowledge"
        return pos, neg

    def normalize_vec(v):
        return v / max(np.linalg.norm(v), 1e-10)

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-10)

    def ndcg_from_scores(scores, doc_ids, qrels_qid, k=10):
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

    model_results = {"model": model_name, "family": family}
    all_features = {}

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
        print(f"\n  Dataset: {ds_name}")
        gc.collect()
        torch.cuda.empty_cache()

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
        centroid = normalize_vec(np.mean(corpus_embs, axis=0))
        neg_generic_emb = embed_model.encode(ds_cfg["neg_generic"], normalize_embeddings=True)

        # IDF
        all_terms = Counter()
        for text in doc_texts:
            all_terms.update(text.lower().split())
        N = len(doc_texts)
        idf = {t: np.log(N / (1 + c)) for t, c in all_terms.items()}

        # Generate per-query targets
        print(f"    Generating per-query targets...")
        pq_pos_texts = []
        pq_neg_texts = []
        for idx, qt in enumerate(query_texts):
            pos_t, neg_t = generate_pos_neg_targets(qt, ds_cfg["domain"])
            pq_pos_texts.append(pos_t)
            pq_neg_texts.append(neg_t)
            if (idx + 1) % 50 == 0:
                print(f"      {idx+1}/{len(query_texts)} targets generated")

        # Encode all unique targets
        unique_targets = list(set(pq_pos_texts + pq_neg_texts))
        print(f"    Encoding {len(unique_targets)} unique targets...")
        tgt_emb_map = {}
        if unique_targets:
            tgt_embs_all = np.array(embed_model.encode(unique_targets, normalize_embeddings=True))
            tgt_emb_map = dict(zip(unique_targets, tgt_embs_all))

        OPERATIONS = [
            "rotate_toward_0.1",
            "rotate_toward_0.2",
            "rotate_away_0.1",
            "rotate_away_0.2",
            "contrastive_asym",
            "multivector_rrf",
            "amplify_0.1",
        ]

        features_list = []
        labels_binary = []
        labels_best_name = []

        for i, (qid, qt) in enumerate(zip(query_ids, query_texts)):
            q = query_embs[i]
            base_scores = q @ corpus_embs.T
            ndcg_base = ndcg_from_scores(base_scores, doc_ids, qrels[qid])

            pos_emb = tgt_emb_map.get(pq_pos_texts[i], normalize_vec(np.random.randn(q.shape[0])))
            neg_emb = tgt_emb_map.get(pq_neg_texts[i], neg_generic_emb)

            op_deltas = {}

            # Rotate toward per-query target
            for alpha in [0.1, 0.2]:
                q_rt = normalize_vec(q + alpha * pos_emb)
                op_deltas[f"rotate_toward_{alpha}"] = ndcg_from_scores(q_rt @ corpus_embs.T, doc_ids, qrels[qid]) - ndcg_base

            # Rotate away per-query target
            for alpha in [0.1, 0.2]:
                q_ra = normalize_vec(q - alpha * pos_emb)
                op_deltas[f"rotate_away_{alpha}"] = ndcg_from_scores(q_ra @ corpus_embs.T, doc_ids, qrels[qid]) - ndcg_base

            # Contrastive with per-query pos AND neg
            q_ca = normalize_vec(q + 0.1 * pos_emb - 0.2 * neg_emb)
            op_deltas["contrastive_asym"] = ndcg_from_scores(q_ca @ corpus_embs.T, doc_ids, qrels[qid]) - ndcg_base

            # Multi-vector RRF with per-query target
            q_mv = normalize_vec(q + 0.1 * pos_emb)
            mv_scores = rrf_fusion_single([base_scores, q_mv @ corpus_embs.T], k=RRF_K)
            op_deltas["multivector_rrf"] = ndcg_from_scores(mv_scores, doc_ids, qrels[qid]) - ndcg_base

            # Amplify
            q_amp = normalize_vec(q + 0.1 * centroid)
            op_deltas["amplify_0.1"] = ndcg_from_scores(q_amp @ corpus_embs.T, doc_ids, qrels[qid]) - ndcg_base

            # Labels
            best_op = max(op_deltas, key=op_deltas.get)
            best_delta = op_deltas[best_op]
            any_positive = any(d > IMPROVEMENT_THRESHOLD for d in op_deltas.values())

            labels_binary.append(1 if any_positive else 0)
            labels_best_name.append(best_op if best_delta > IMPROVEMENT_THRESHOLD else "none")

            # Features (11 total — v2 features + per-query target features)
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
            # NEW: pos-neg separation (how different are the two targets?)
            pos_neg_separation = float(pos_emb @ neg_emb)

            features_list.append([
                query_length, specificity, dist_to_centroid,
                embedding_norm, isotropy_local, query_pos_sim,
                query_neg_sim, centroid_alignment, target_orthogonality,
                pos_neg_separation,
                # Derived: adaptive alpha value for this query
                0.2 * (1 - query_pos_sim) ** 2,
            ])

        X = np.array(features_list)
        y_bin = np.array(labels_binary)

        all_features[ds_name] = (X, y_bin)

        pos_rate = float(np.mean(y_bin))
        print(f"    Positive rate (any op, per-query targets): {pos_rate:.1%}")
        print(f"    Best op distribution: {Counter(labels_best_name).most_common()}")

        if len(np.unique(y_bin)) < 2:
            model_results[ds_name] = {"error": "single_class", "positive_rate": pos_rate}
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        feature_names = [
            "query_length", "specificity", "dist_to_centroid",
            "embedding_norm", "isotropy_local", "query_pos_sim",
            "query_neg_sim", "centroid_alignment", "target_orthogonality",
            "pos_neg_separation", "adaptive_alpha_value",
        ]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classifiers = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

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
                print(f"    {clf_name}: Acc={np.mean(cv_acc):.3f} F1={np.mean(cv_f1):.3f} P={np.mean(cv_prec):.3f} R={np.mean(cv_rec):.3f}")
            except Exception as e:
                binary_results[clf_name] = {"error": str(e)}

        # Feature importances
        best_clf_name = max(binary_results, key=lambda k: binary_results[k].get("f1", 0))
        best_clf = classifiers[best_clf_name]
        best_clf.fit(X_scaled, y_bin)
        if hasattr(best_clf, 'feature_importances_'):
            importances = dict(zip(feature_names, [round(float(c), 4) for c in best_clf.feature_importances_]))
        elif hasattr(best_clf, 'coef_'):
            importances = dict(zip(feature_names, [round(float(abs(c)), 4) for c in best_clf.coef_[0]]))
        else:
            importances = {}

        ds_result = {
            "n_queries": len(query_ids),
            "positive_rate_perquery_targets": round(pos_rate, 4),
            "best_op_distribution": dict(Counter(labels_best_name).most_common()),
            "binary_classifiers": binary_results,
            "best_binary_classifier": best_clf_name,
            "feature_importances": importances,
            "top_feature": max(importances, key=importances.get) if importances else "unknown",
        }
        model_results[ds_name] = ds_result

    # Cross-dataset transfer
    if "trec-covid" in all_features and "scifact" in all_features:
        X_train, y_train = all_features["trec-covid"]
        X_test, y_test = all_features["scifact"]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            model_results["cross_dataset_transfer"] = {
                "train": "trec-covid", "test": "scifact",
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            }
            print(f"\n  Transfer trec-covid→scifact: Acc={accuracy_score(y_test, y_pred):.3f} F1={f1_score(y_test, y_pred, zero_division=0):.3f}")

    safe = model_name.replace("/", "_")
    out_path = f"/results/steer_classifier_v3/{safe}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {out_path}")
    return model_results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("  STEER — Classifier v3: Per-Query LLM Targets (3 models x 5 DS)")
    print("=" * 70)
    results = list(run_classifier_v3.starmap([(n, f) for n, f in MODELS]))

    print("\n" + "=" * 70)
    print("  COMPARISON: v2 (generic targets) vs v3 (per-query targets)")
    print("=" * 70)
    for r in results:
        m = r['model'].split('/')[-1]
        for ds in ['scifact', 'arguana', 'nfcorpus', 'fiqa', 'trec-covid']:
            if ds not in r or 'error' in r[ds]:
                continue
            d = r[ds]
            best = d.get('best_binary_classifier', '?')
            f1 = d.get('binary_classifiers', {}).get(best, {}).get('f1', 0)
            pos = d.get('positive_rate_perquery_targets', 0)
            print(f"  {m:22s} {ds:12s} pos_rate={pos:.1%} best_F1={f1:.3f} ({best})")

    print("\n  DONE!")
