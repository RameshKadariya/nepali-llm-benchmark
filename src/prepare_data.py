"""
Phase 1: Dataset acquisition and freezing.

Downloads open-license source datasets, cleans them, and freezes fixed
evaluation samples (seed 42) into data/frozen/ as JSONL. Every experiment
in this repository reads only from data/frozen/, never from live sources,
so all results are exactly reproducible.

Sources:
  1. facebook/belebele (CC-BY-SA 4.0)
     - Reading comprehension MCQs over FLORES-200 passages.
     - npi_Deva (Nepali) and eng_Latn (English), aligned by 'link' +
       'question_number', enabling paired same-content comparison.
  2. Shushant/NepaliSentiment (native Nepali social media text)
     - 3-class sentiment: 0 = negative, 1 = positive, 2 = neutral.
     - Label semantics verified by manual inspection and native speaker
       validation (see data/validation/).

Derived sets frozen by this script:
  qa_ne.jsonl / qa_en.jsonl        200 paired MCQ items per language
  translation_pairs.jsonl          150 parallel ne<->en FLORES passages
  sentiment_ne.jsonl               210 items, 70 per class
  retrieval_corpus_ne.jsonl        all unique Nepali passages (corpus)
  retrieval_queries.jsonl          200 queries (ne + en versions) with
                                   gold passage ids

Usage: python src/prepare_data.py
"""

import hashlib
import json
import os
import random
import re

from datasets import load_dataset

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
VALID = os.path.join(ROOT, "data", "validation")
os.makedirs(FROZEN, exist_ok=True)
os.makedirs(VALID, exist_ok=True)

N_QA = 200
N_TRANSLATION = 150
N_SENTIMENT_PER_CLASS = 70
N_RETRIEVAL_QUERIES = 200
N_VALIDATION = 30

SENTIMENT_LABEL_MAP = {"0": "negative", "1": "positive", "2": "neutral"}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5} rows -> {os.path.relpath(path, ROOT)}")


def passage_id(link):
    return "p" + hashlib.sha1(link.encode()).hexdigest()[:10]


def main():
    rng = random.Random(SEED)

    # ------------------------------------------------------------------
    # 1. Belebele: load aligned Nepali and English splits
    # ------------------------------------------------------------------
    print("Loading Belebele (npi_Deva, eng_Latn)...")
    b_ne = load_dataset("facebook/belebele", "npi_Deva", split="test")
    b_en = load_dataset("facebook/belebele", "eng_Latn", split="test")

    en_index = {(r["link"], r["question_number"]): r for r in b_en}

    paired = []
    for r_ne in b_ne:
        key = (r_ne["link"], r_ne["question_number"])
        if key in en_index:
            paired.append((r_ne, en_index[key]))
    assert len(paired) == 900, f"expected 900 aligned items, got {len(paired)}"
    print(f"  aligned QA pairs: {len(paired)}")

    # ------------------------------------------------------------------
    # 2. QA task: sample 200 paired items
    # ------------------------------------------------------------------
    qa_sample = rng.sample(paired, N_QA)
    qa_ne_rows, qa_en_rows = [], []
    for i, (ne, en) in enumerate(qa_sample):
        base = {
            "id": f"qa{i:04d}",
            "passage_id": passage_id(ne["link"]),
            "question_number": ne["question_number"],
            "correct_answer_num": int(ne["correct_answer_num"]),
        }
        qa_ne_rows.append({**base, "lang": "ne",
                           "passage": ne["flores_passage"],
                           "question": ne["question"],
                           "options": [ne["mc_answer1"], ne["mc_answer2"],
                                       ne["mc_answer3"], ne["mc_answer4"]]})
        qa_en_rows.append({**base, "lang": "en",
                           "passage": en["flores_passage"],
                           "question": en["question"],
                           "options": [en["mc_answer1"], en["mc_answer2"],
                                       en["mc_answer3"], en["mc_answer4"]]})
    write_jsonl(os.path.join(FROZEN, "qa_ne.jsonl"), qa_ne_rows)
    write_jsonl(os.path.join(FROZEN, "qa_en.jsonl"), qa_en_rows)

    # ------------------------------------------------------------------
    # 3. Parallel passages (unique by link) for translation + retrieval
    # ------------------------------------------------------------------
    uniq = {}
    for ne, en in paired:
        pid = passage_id(ne["link"])
        if pid not in uniq:
            uniq[pid] = {"passage_id": pid,
                         "ne": ne["flores_passage"],
                         "en": en["flores_passage"]}
    passages = sorted(uniq.values(), key=lambda x: x["passage_id"])
    print(f"  unique parallel passages: {len(passages)}")

    trans_sample = rng.sample(passages, N_TRANSLATION)
    trans_rows = [{"id": f"tr{i:04d}", **p} for i, p in enumerate(trans_sample)]
    write_jsonl(os.path.join(FROZEN, "translation_pairs.jsonl"), trans_rows)

    # Retrieval corpus: every unique Nepali passage
    corpus_rows = [{"passage_id": p["passage_id"], "text_ne": p["ne"]}
                   for p in passages]
    write_jsonl(os.path.join(FROZEN, "retrieval_corpus_ne.jsonl"), corpus_rows)

    # Retrieval queries: questions whose gold passage is in the corpus.
    # One question per passage to avoid corpus-frequency bias, then sample.
    seen_pid = set()
    query_pool = []
    for ne, en in paired:
        pid = passage_id(ne["link"])
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        query_pool.append({"gold_passage_id": pid,
                           "query_ne": ne["question"],
                           "query_en": en["question"]})
    q_sample = rng.sample(query_pool, N_RETRIEVAL_QUERIES)
    q_rows = [{"id": f"rq{i:04d}", **q} for i, q in enumerate(q_sample)]
    write_jsonl(os.path.join(FROZEN, "retrieval_queries.jsonl"), q_rows)

    # ------------------------------------------------------------------
    # 4. Sentiment: clean, dedupe, stratified sample 70 per class
    # ------------------------------------------------------------------
    print("Loading Shushant/NepaliSentiment...")
    sent = load_dataset("Shushant/NepaliSentiment", split="train")
    devanagari = re.compile(r"[\u0900-\u097F]")
    by_class = {"negative": [], "positive": [], "neutral": []}
    seen_text = set()
    for r in sent:
        lab = str(r["label"]).strip()
        if lab not in SENTIMENT_LABEL_MAP:
            continue  # drop the handful of malformed labels
        text = (r["text"] or "").strip()
        if len(text) < 15 or len(text) > 500:
            continue
        if len(devanagari.findall(text)) < 10:
            continue  # require substantially Devanagari text
        if text in seen_text:
            continue
        seen_text.add(text)
        by_class[SENTIMENT_LABEL_MAP[lab]].append(text)

    for k, v in by_class.items():
        print(f"  sentiment pool [{k}]: {len(v)}")

    sent_rows = []
    for cls in ["negative", "positive", "neutral"]:
        for text in rng.sample(by_class[cls], N_SENTIMENT_PER_CLASS):
            sent_rows.append({"text": text, "label": cls})
    rng.shuffle(sent_rows)
    sent_rows = [{"id": f"sn{i:04d}", **r} for i, r in enumerate(sent_rows)]
    write_jsonl(os.path.join(FROZEN, "sentiment_ne.jsonl"), sent_rows)

    # ------------------------------------------------------------------
    # 5. Native speaker validation sample
    # ------------------------------------------------------------------
    val_rows = []
    for r in rng.sample(sent_rows, 15):
        val_rows.append({"task": "sentiment", "id": r["id"],
                         "text": r["text"], "assigned_label": r["label"],
                         "native_speaker_verdict": ""})
    for r in rng.sample(qa_ne_rows, 10):
        val_rows.append({"task": "qa", "id": r["id"],
                         "question": r["question"],
                         "options": r["options"],
                         "correct_option": r["correct_answer_num"],
                         "native_speaker_verdict": ""})
    for r in rng.sample(trans_rows, 5):
        val_rows.append({"task": "translation", "id": r["id"],
                         "nepali": r["ne"], "english": r["en"],
                         "native_speaker_verdict": ""})
    write_jsonl(os.path.join(VALID, "validation_sample.jsonl"), val_rows)

    print("\nAll frozen sets written. Do not regenerate without bumping "
          "the dataset version in the paper.")


if __name__ == "__main__":
    main()
