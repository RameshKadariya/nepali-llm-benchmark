"""
Task 3 (API models): Sentiment classification of native Nepali text.

210 frozen items (70 negative, 70 positive, 70 neutral) of authentic
Nepali social media text. The model receives a Nepali instruction and
must answer with one word. Parsing accepts the Nepali label words and
their English equivalents. Temperature 0.

Metrics: accuracy, macro-F1, per-class recall, abstention count.

Usage:
  python src/task3_sentiment_api.py <model_key> [max_items]
  python src/task3_sentiment_api.py merge
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
TAB_OUT = os.path.join(ROOT, "results", "tables")
PROG = os.path.join(RAW_OUT, "task3_progress")
os.makedirs(PROG, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task2_qa_api import API_MODELS, call_model, run_parallel  # noqa: E402

PROMPT = ("तलको नेपाली पाठको भावना वर्गीकरण गर्नुहोस्। "
          "उत्तरमा केवल एउटा शब्द लेख्नुहोस्: "
          "सकारात्मक, नकारात्मक वा तटस्थ।\n\nपाठ: {text}\n\nभावना:")

LABEL_WORDS = {
    "positive": ["सकारात्मक", "positive"],
    "negative": ["नकारात्मक", "negative"],
    "neutral": ["तटस्थ", "neutral"],
}


def parse_label(text):
    t = text.strip().lower()
    hits = [lab for lab, words in LABEL_WORDS.items()
            if any(w in t for w in words)]
    return hits[0] if len(hits) == 1 else None


def load_items():
    with open(os.path.join(FROZEN, "sentiment_ne.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(model_key, max_items):
    provider, model_id, name = API_MODELS[model_key]
    items = load_items()
    prog_path = os.path.join(PROG, f"api_{model_key}.jsonl")
    done = set()
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [it for it in items if it["id"] not in done]
    print(f"=== {name} [sentiment] done={len(done)} todo={len(todo)}")

    def make_record(it):
        text = call_model(provider, model_id, PROMPT.format(text=it["text"]))
        pred = parse_label(text)
        return {"id": it["id"], "pred": pred, "gold": it["label"],
                "correct": pred == it["label"], "raw": text[:120]}

    workers = 3 if provider == "gemini" else 5
    run_parallel(todo[:max_items], make_record, prog_path, workers)
    print("window complete")


def macro_f1(recs, labels=("negative", "positive", "neutral")):
    f1s = []
    per_class = {}
    for lab in labels:
        tp = sum(1 for r in recs if r["pred"] == lab and r["gold"] == lab)
        fp = sum(1 for r in recs if r["pred"] == lab and r["gold"] != lab)
        fn = sum(1 for r in recs if r["pred"] != lab and r["gold"] == lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rcl = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0.0
        f1s.append(f1)
        per_class[lab] = {"precision": round(p, 4), "recall": round(rcl, 4),
                          "f1": round(f1, 4)}
    return round(sum(f1s) / len(f1s), 4), per_class


def merge():
    import csv
    rows = []
    for key, (provider, model_id, name) in API_MODELS.items():
        p = os.path.join(PROG, f"api_{key}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        if not recs:
            continue
        mf1, per_class = macro_f1(recs)
        rows.append({"model": name, "api_id": model_id,
                     "accuracy": round(sum(r["correct"] for r in recs)
                                       / len(recs), 4),
                     "macro_f1": mf1, "n": len(recs),
                     "abstain": sum(r["pred"] is None for r in recs),
                     "per_class": per_class})
    with open(os.path.join(RAW_OUT, "task3_sentiment.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    with open(os.path.join(TAB_OUT, "task3_sentiment.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "api_id", "accuracy",
                                          "macro_f1", "n", "abstain"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    for r in rows:
        print(r["model"], "acc:", r["accuracy"], "macroF1:", r["macro_f1"],
              "abstain:", r["abstain"])


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        run(sys.argv[1],
            int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9)
