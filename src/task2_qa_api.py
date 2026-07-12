"""
Task 2 (API models): Belebele reading comprehension, Nepali vs English.

API-served chat models are evaluated generatively: the model sees the
passage, question, and four numbered options, and must reply with the
number of the correct option. Temperature 0. The first digit 1-4 in
the reply is taken as the prediction; replies with no parseable digit
are recorded as abstentions (counted as incorrect, and reported).

Same 200 paired items as the local log-likelihood evaluation.

API keys are read from environment variables GROQ_API_KEY and
GEMINI_API_KEY and are never written to disk.

Usage:
  python src/task2_qa_api.py <model_key> <lang: ne|en> [max_items]
  python src/task2_qa_api.py merge

Progress: results/raw/task2_progress/api_<model_key>_<lang>.jsonl
"""

import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
TAB_OUT = os.path.join(ROOT, "results", "tables")
PROG = os.path.join(RAW_OUT, "task2_progress")
os.makedirs(PROG, exist_ok=True)

# model_key -> (provider, api model id, display name)
API_MODELS = {
    "llama70b": ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B"),
    "llama8b": ("groq", "llama-3.1-8b-instant", "Llama 3.1 8B"),
    "scout17b": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct",
                 "Llama 4 Scout 17B"),
    "gptoss120b": ("groq", "openai/gpt-oss-120b", "GPT-OSS 120B"),
    "qwen32b": ("groq", "qwen/qwen3-32b", "Qwen3 32B"),
    "geminiflash": ("gemini", "gemini-flash-latest", "Gemini 3.5 Flash"),
    "geminilite": ("gemini", "gemini-flash-lite-latest",
                   "Gemini 3.1 Flash Lite"),
}

PROMPT = {
    "ne": ("तलको अनुच्छेद पढेर प्रश्नको सही उत्तर छान्नुहोस्। "
           "केवल सही विकल्पको नम्बर (1, 2, 3 वा 4) मात्र लेख्नुहोस्।\n\n"
           "अनुच्छेद: {passage}\n\nप्रश्न: {question}\n\n"
           "विकल्पहरू:\n1. {o1}\n2. {o2}\n3. {o3}\n4. {o4}\n\nउत्तर:"),
    "en": ("Read the passage below and choose the correct answer to the "
           "question. Reply with only the number of the correct option "
           "(1, 2, 3 or 4).\n\n"
           "Passage: {passage}\n\nQuestion: {question}\n\n"
           "Options:\n1. {o1}\n2. {o2}\n3. {o3}\n4. {o4}\n\nAnswer:"),
}


def post_json(url, headers, payload, retries=6):
    body = json.dumps(payload).encode()
    headers = {**headers, "User-Agent": "nepali-llm-benchmark/1.0"}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = min(60, 2 ** attempt * 3)
                print(f"  HTTP {e.code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("retries exhausted")


def call_model(provider, model_id, prompt):
    if provider == "groq":
        key = os.environ["GROQ_API_KEY"]
        payload = {"model": model_id, "temperature": 0, "max_tokens": 2048,
                   "messages": [{"role": "user", "content": prompt}]}
        if model_id.startswith("qwen/"):
            payload["reasoning_effort"] = "none"
        d = post_json("https://api.groq.com/openai/v1/chat/completions",
                      {"Authorization": f"Bearer {key}",
                       "Content-Type": "application/json"}, payload)
        return d["choices"][0]["message"]["content"] or ""
    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id}:generateContent?key={key}")
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0,
                                        "maxOutputTokens": 2048}}
        d = post_json(url, {"Content-Type": "application/json"}, payload)
        try:
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""
    raise ValueError(provider)


DEVANAGARI_DIGITS = {"\u0967": 1, "\u0968": 2, "\u0969": 3, "\u096a": 4}


def parse_answer(text):
    """First option number in the reply; accepts ASCII 1-4 and
    Devanagari numerals (some models answer in the script of the
    question)."""
    m = re.search(r"[1-4\u0967-\u096a]", text)
    if not m:
        return None
    ch = m.group()
    return DEVANAGARI_DIGITS.get(ch, int(ch) if ch.isdigit() else None)


def load_items(lang):
    with open(os.path.join(FROZEN, f"qa_{lang}.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(model_key, lang, max_items):
    provider, model_id, name = API_MODELS[model_key]
    items = load_items(lang)
    prog_path = os.path.join(PROG, f"api_{model_key}_{lang}.jsonl")
    done = set()
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [it for it in items if it["id"] not in done]
    print(f"=== {name} [{lang}] done={len(done)} todo={len(todo)}")

    def make_record(it):
        prompt = PROMPT[lang].format(
            passage=it["passage"], question=it["question"],
            o1=it["options"][0], o2=it["options"][1],
            o3=it["options"][2], o4=it["options"][3])
        text = call_model(provider, model_id, prompt)
        pred = parse_answer(text)
        return {"id": it["id"], "pred": pred,
                "gold": it["correct_answer_num"],
                "correct": pred == it["correct_answer_num"],
                "raw": text[:200]}

    workers = 3 if provider == "gemini" else 5
    run_parallel(todo[:max_items], make_record, prog_path, workers)
    print("window complete")




def run_parallel(todo, make_record, prog_path, workers=5):
    """Process items concurrently, appending JSON records as they finish."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()
    done_count = 0
    with open(prog_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(make_record, it): it for it in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    done_count += 1
                    if done_count % 25 == 0:
                        print(f"  +{done_count}/{len(todo)}")


def merge():
    import csv
    rows = []
    for key, (provider, model_id, name) in API_MODELS.items():
        row = {"model": name, "api_id": model_id, "provider": provider,
               "method": "generation"}
        for lang in ["ne", "en"]:
            p = os.path.join(PROG, f"api_{key}_{lang}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                recs = [json.loads(line) for line in f]
            if not recs:
                continue
            row[f"acc_{lang}"] = round(
                sum(r["correct"] for r in recs) / len(recs), 4)
            row[f"n_{lang}"] = len(recs)
            row[f"abstain_{lang}"] = sum(r["pred"] is None for r in recs)
        if "acc_ne" in row and "acc_en" in row:
            row["drop_en_to_ne"] = round(row["acc_en"] - row["acc_ne"], 4)
        if "acc_ne" in row or "acc_en" in row:
            rows.append(row)
    with open(os.path.join(RAW_OUT, "task2_qa_api.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k != "model", k))
    with open(os.path.join(TAB_OUT, "task2_qa_api.csv"), "w",
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
