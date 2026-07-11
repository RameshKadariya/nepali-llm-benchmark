"""
Task 5: Embedding retrieval quality on Nepali.

Simulates the retrieval stage of a Nepali RAG system. The corpus is
488 unique Nepali passages; queries are 200 questions written against
those passages (gold passage known). Two settings per model:

  mono : Nepali query  -> Nepali corpus
  cross: English query -> Nepali corpus (cross-lingual alignment)

Metrics: Recall@1, Recall@5, Recall@10, MRR@10.

All models run locally on CPU. An English-only model (all-MiniLM-L6-v2)
is included deliberately as a negative control: it shows what happens
when a popular default embedding model meets Devanagari.

Outputs:
  results/raw/task5_retrieval.json
  results/tables/task5_retrieval.csv

Usage: python src/task5_retrieval.py
"""

import gc
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
TAB_OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(RAW_OUT, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)

# (display name, HF id, query prefix, passage prefix, multilingual?)
# E5 models require the documented "query: " / "passage: " prefixes.
MODELS = [
    ("multilingual-e5-small", "intfloat/multilingual-e5-small",
     "query: ", "passage: ", True),
    ("multilingual-e5-base", "intfloat/multilingual-e5-base",
     "query: ", "passage: ", True),
    ("paraphrase-multilingual-MiniLM-L12-v2",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "", "", True),
    ("LaBSE", "sentence-transformers/LaBSE", "", "", True),
    ("BGE-M3", "BAAI/bge-m3", "", "", True),
    ("all-MiniLM-L6-v2 (English-only control)",
     "sentence-transformers/all-MiniLM-L6-v2", "", "", False),
]


def load_jsonl(name):
    with open(os.path.join(FROZEN, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(sims, gold_idx):
    """sims: (n_queries, n_docs); gold_idx: list of gold doc indices."""
    order = np.argsort(-sims, axis=1)
    r1 = r5 = r10 = 0
    mrr = 0.0
    ranks = []
    for i, g in enumerate(gold_idx):
        rank = int(np.where(order[i] == g)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
        if rank <= 10:
            mrr += 1.0 / rank
    n = len(gold_idx)
    return {"recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4),
            "recall@10": round(r10 / n, 4), "mrr@10": round(mrr / n, 4),
            "ranks": ranks}


def main():
    from sentence_transformers import SentenceTransformer

    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    parts_dir = os.path.join(RAW_OUT, "task5_parts")
    os.makedirs(parts_dir, exist_ok=True)

    corpus = load_jsonl("retrieval_corpus_ne.jsonl")
    queries = load_jsonl("retrieval_queries.jsonl")
    pid_to_idx = {c["passage_id"]: i for i, c in enumerate(corpus)}
    gold_idx = [pid_to_idx[q["gold_passage_id"]] for q in queries]
    docs = [c["text_ne"] for c in corpus]
    print(f"Corpus: {len(docs)} passages | Queries: {len(queries)}")

    all_results = []
    for mi, (name, hf_id, qpre, ppre, multi) in enumerate(MODELS):
        if only is not None and mi != only:
            continue
        print(f"\n=== {name}")
        try:
            model = SentenceTransformer(hf_id, device="cpu")
            model.max_seq_length = min(model.max_seq_length or 512, 512)
        except Exception as e:
            print(f"SKIP: {str(e)[:120]}")
            continue

        doc_emb = model.encode([ppre + d for d in docs],
                               batch_size=16, show_progress_bar=False,
                               normalize_embeddings=True)
        row = {"model": name, "hf_id": hf_id,
               "multilingual_by_design": multi}
        per_query = {}
        for setting, key in [("mono", "query_ne"), ("cross", "query_en")]:
            q_emb = model.encode([qpre + q[key] for q in queries],
                                 batch_size=16, show_progress_bar=False,
                                 normalize_embeddings=True)
            sims = q_emb @ doc_emb.T
            m = evaluate(sims, gold_idx)
            per_query[setting] = m.pop("ranks")
            for k, v in m.items():
                row[f"{setting}_{k}"] = v
            print(f"  {setting:5s} R@1={row[f'{setting}_recall@1']:.3f} "
                  f"R@5={row[f'{setting}_recall@5']:.3f} "
                  f"R@10={row[f'{setting}_recall@10']:.3f} "
                  f"MRR@10={row[f'{setting}_mrr@10']:.3f}")

        rec = {"summary": row, "per_query_ranks": per_query,
               "query_ids": [q["id"] for q in queries]}
        all_results.append(rec)
        with open(os.path.join(parts_dir, f"{mi:02d}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        del model, doc_emb
        gc.collect()

    # merge all completed parts
    all_results = []
    for fn in sorted(os.listdir(parts_dir)):
        with open(os.path.join(parts_dir, fn), encoding="utf-8") as f:
            all_results.append(json.load(f))

    with open(os.path.join(RAW_OUT, "task5_retrieval.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    import csv
    fields = list(all_results[0]["summary"].keys())
    with open(os.path.join(TAB_OUT, "task5_retrieval.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow(r["summary"])
    print("\nWrote results/raw/task5_retrieval.json and "
          "results/tables/task5_retrieval.csv")


if __name__ == "__main__":
    main()
