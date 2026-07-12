"""
Task 4 (API models): Machine translation quality, ne->en and en->ne.

150 frozen parallel FLORES passages (via Belebele, CC-BY-SA 4.0).
The model translates each passage; outputs are stored verbatim and
scored with chrF++ (primary; robust for Devanagari) and BLEU
(secondary; flores200 sentencepiece tokenization when scoring Nepali
output). Temperature 0.

Usage:
  python src/task4_translation_api.py <model_key> <dir: ne2en|en2ne> [max]
  python src/task4_translation_api.py merge
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
PROG = os.path.join(RAW_OUT, "task4_progress")
os.makedirs(PROG, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task2_qa_api import API_MODELS, call_model, run_parallel  # noqa: E402

PROMPT = {
    "ne2en": ("Translate the following Nepali text to English. "
              "Output only the English translation, nothing else.\n\n"
              "{text}"),
    "en2ne": ("Translate the following English text to Nepali. "
              "Output only the Nepali translation, nothing else.\n\n"
              "{text}"),
}


def load_pairs():
    with open(os.path.join(FROZEN, "translation_pairs.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(model_key, direction, max_items):
    provider, model_id, name = API_MODELS[model_key]
    pairs = load_pairs()
    src_key = "ne" if direction == "ne2en" else "en"
    prog_path = os.path.join(PROG, f"api_{model_key}_{direction}.jsonl")
    done = set()
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [p for p in pairs if p["id"] not in done]
    print(f"=== {name} [{direction}] done={len(done)} todo={len(todo)}")

    def make_record(p):
        out = call_model(provider, model_id,
                         PROMPT[direction].format(text=p[src_key]))
        return {"id": p["id"], "hypothesis": out.strip()}

    workers = 3 if provider == "gemini" else 5
    run_parallel(todo[:max_items], make_record, prog_path, workers)
    print("window complete")


def score(direction, hyp_by_id, pairs):
    import sacrebleu
    ref_key = "en" if direction == "ne2en" else "ne"
    ids = [p["id"] for p in pairs if p["id"] in hyp_by_id]
    hyps = [hyp_by_id[i] for i in ids]
    refs = [[p[ref_key] for p in pairs if p["id"] in hyp_by_id]]
    chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2)  # chrF++
    tok = "flores200" if direction == "en2ne" else "13a"
    try:
        bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=tok)
    except Exception:
        bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize="char")
        tok = "char"
    return {"chrf++": round(chrf.score, 2), "bleu": round(bleu.score, 2),
            "bleu_tokenize": tok, "n": len(hyps)}


def merge():
    import csv
    pairs = load_pairs()
    rows = []
    for key, (provider, model_id, name) in API_MODELS.items():
        row = {"model": name, "api_id": model_id}
        got = False
        for direction in ["ne2en", "en2ne"]:
            p = os.path.join(PROG, f"api_{key}_{direction}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                hyp = {json.loads(l)["id"]: json.loads(l)["hypothesis"]
                       for l in f}
            if not hyp:
                continue
            m = score(direction, hyp, pairs)
            for k, v in m.items():
                row[f"{direction}_{k}"] = v
            got = True
        if got:
            rows.append(row)
    with open(os.path.join(RAW_OUT, "task4_translation.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k != "model", k))
    with open(os.path.join(TAB_OUT, "task4_translation.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        run(sys.argv[1], sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9)
