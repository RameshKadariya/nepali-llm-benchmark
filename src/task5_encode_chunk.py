"""
Resumable chunked encoding for heavy embedding models on CPU.

Encodes the Task 5 corpus and queries in chunks, caching progress to
results/raw/task5_cache/<tag>.npy files so the run can resume across
interrupted sessions. When everything is encoded, computes the same
metrics as task5_retrieval.py and writes the part file.

Usage: python src/task5_encode_chunk.py <model_index> <chunk_size>
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
CACHE = os.path.join(RAW_OUT, "task5_cache")
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task5_retrieval import MODELS, evaluate, load_jsonl  # noqa: E402


def encode_resumable(model, tag, texts, chunk):
    done = sorted(f for f in os.listdir(CACHE)
                  if f.startswith(tag) and f.endswith(".npy"))
    start = sum(np.load(os.path.join(CACHE, f)).shape[0] for f in done)
    idx = len(done)
    while start < len(texts):
        batch = texts[start:start + chunk]
        emb = model.encode(batch, batch_size=8, show_progress_bar=False,
                           normalize_embeddings=True)
        np.save(os.path.join(CACHE, f"{tag}_{idx:03d}.npy"), emb)
        start += len(batch)
        idx += 1
        print(f"  {tag}: {start}/{len(texts)} encoded")
    parts = sorted(f for f in os.listdir(CACHE)
                   if f.startswith(tag) and f.endswith(".npy"))
    return np.vstack([np.load(os.path.join(CACHE, f)) for f in parts])


def main():
    mi = int(sys.argv[1])
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    name, hf_id, qpre, ppre, multi = MODELS[mi]
    print(f"=== {name} (chunked, chunk={chunk})")

    corpus = load_jsonl("retrieval_corpus_ne.jsonl")
    queries = load_jsonl("retrieval_queries.jsonl")
    pid_to_idx = {c["passage_id"]: i for i, c in enumerate(corpus)}
    gold_idx = [pid_to_idx[q["gold_passage_id"]] for q in queries]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hf_id, device="cpu")
    model.max_seq_length = min(model.max_seq_length or 512, 512)

    doc_emb = encode_resumable(model, f"m{mi}_docs",
                               [ppre + c["text_ne"] for c in corpus], chunk)
    qne_emb = encode_resumable(model, f"m{mi}_qne",
                               [qpre + q["query_ne"] for q in queries], chunk)
    qen_emb = encode_resumable(model, f"m{mi}_qen",
                               [qpre + q["query_en"] for q in queries], chunk)

    row = {"model": name, "hf_id": hf_id, "multilingual_by_design": multi}
    per_query = {}
    for setting, q_emb in [("mono", qne_emb), ("cross", qen_emb)]:
        m = evaluate(q_emb @ doc_emb.T, gold_idx)
        per_query[setting] = m.pop("ranks")
        for k, v in m.items():
            row[f"{setting}_{k}"] = v
        print(f"  {setting:5s}", {k: v for k, v in row.items()
                                  if k.startswith(setting)})

    parts_dir = os.path.join(RAW_OUT, "task5_parts")
    os.makedirs(parts_dir, exist_ok=True)
    with open(os.path.join(parts_dir, f"{mi:02d}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"summary": row, "per_query_ranks": per_query,
                   "query_ids": [q["id"] for q in queries]},
                  f, ensure_ascii=False, indent=1)
    print("part saved")


if __name__ == "__main__":
    main()
