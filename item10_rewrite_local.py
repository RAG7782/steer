"""
Item 10: Local rewriting step using Ollama gemma3:12b.

Runs LOCALLY (not on Modal) — rewrites 100 SciFact docs in plain language.
Output: results/item10_preprocessing/rewritten_docs.json

After this completes, run the evaluation on Modal:
    modal run modal_benchmarks.py --items 10 --rewritten-docs results/item10_preprocessing/rewritten_docs.json
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
DATA_DIR = Path("data/beir/scifact")
RESULTS_DIR = Path("results/item10_preprocessing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_DOCS = 100

REWRITE_PROMPT = """Rewrite this scientific abstract replacing ALL jargon, abbreviations, method names, and technical terms with plain conceptual descriptions. Keep the same meaning but use everyday language. Be concise — output only the rewritten text, nothing else.

Original:
{text}

Rewritten:"""


def rewrite_with_ollama(text: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": REWRITE_PROMPT.format(text=text),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512},
            }, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt == max_retries:
                print(f"    FAILED: {e}")
                return text
            time.sleep(2)


def main():
    np.random.seed(42)

    print("Loading SciFact...")
    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR)).load(split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                 for d in doc_ids]

    # Select docs with relevance judgments first
    relevant_docs = set()
    for qid, rels in qrels.items():
        for did, score in rels.items():
            if score > 0:
                relevant_docs.add(did)

    selected_idx = []
    for i, did in enumerate(doc_ids):
        if did in relevant_docs and len(selected_idx) < N_DOCS:
            selected_idx.append(i)
    remaining = [i for i in range(len(doc_ids)) if i not in selected_idx]
    np.random.shuffle(remaining)
    selected_idx.extend(remaining[:max(0, N_DOCS - len(selected_idx))])
    selected_idx = sorted(selected_idx[:N_DOCS])

    print(f"Selected {len(selected_idx)} docs ({sum(1 for i in selected_idx if doc_ids[i] in relevant_docs)} with relevance)")

    # Load cache if exists
    cache_file = RESULTS_DIR / "rewritten_docs.json"
    rewritten_texts = {}
    if cache_file.exists():
        with open(cache_file) as f:
            rewritten_texts = json.load(f)
        print(f"Loaded {len(rewritten_texts)} cached rewrites")

    # Rewrite
    t0 = time.time()
    for count, idx in enumerate(selected_idx):
        did = doc_ids[idx]
        if did in rewritten_texts:
            continue
        original = doc_texts[idx]
        rewritten = rewrite_with_ollama(original)
        rewritten_texts[did] = rewritten

        elapsed = time.time() - t0
        done = count + 1
        remaining_count = len(selected_idx) - done
        rate = done / elapsed if elapsed > 0 else 0
        eta = remaining_count / rate if rate > 0 else 0

        if done % 5 == 0 or count == 0:
            print(f"  [{done}/{len(selected_idx)}] {rate:.1f} docs/s  ETA: {eta:.0f}s  "
                  f"orig: {len(original)}→{len(rewritten)} chars")

        # Save every 10 docs
        if done % 10 == 0:
            with open(cache_file, "w") as f:
                json.dump(rewritten_texts, f, indent=2)

    # Final save
    with open(cache_file, "w") as f:
        json.dump(rewritten_texts, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! {len(rewritten_texts)} docs rewritten in {elapsed:.0f}s")
    print(f"Saved to {cache_file}")
    print(f"\nNext: modal run modal_benchmarks.py --items 10 --rewritten-docs {cache_file}")


if __name__ == "__main__":
    main()
