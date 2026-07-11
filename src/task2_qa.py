"""
Task 2: Reading comprehension (Belebele) in Nepali vs English.

Method: multiple-choice scoring by length-normalized log-likelihood,
the same protocol used by lm-evaluation-harness for Belebele-style
tasks. For each item, the model scores the four candidate answers as
continuations of a prompt containing the passage and question; the
highest-scoring option is the prediction. No sampling, fully
deterministic, and usable with base or instruct models alike.

The same 200 items are evaluated in both languages (paired design),
so the English-to-Nepali accuracy drop is measured within-model on
identical content.

Runs are resumable: per-item results append to a JSONL progress file
per (model, language) and completed items are skipped on restart.

Usage:
  python src/task2_qa.py <model_index> <lang: ne|en> [max_items]

Merge all progress into final results:
  python src/task2_qa.py merge
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
TAB_OUT = os.path.join(ROOT, "results", "tables")
PROG = os.path.join(RAW_OUT, "task2_progress")
os.makedirs(PROG, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"),
    ("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
]

PROMPT = {
    "ne": ("तलको अनुच्छेद पढेर प्रश्नको सही उत्तर छान्नुहोस्।\n\n"
           "अनुच्छेद: {passage}\n\nप्रश्न: {question}\n\nउत्तर:"),
    "en": ("Read the passage below and choose the correct answer to "
           "the question.\n\nPassage: {passage}\n\nQuestion: {question}"
           "\n\nAnswer:"),
}


def load_items(lang):
    with open(os.path.join(FROZEN, f"qa_{lang}.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def score_options(model, tok, prompt, options, device="cpu"):
    """Length-normalized log-likelihood of each option continuation.

    All four options are scored in a single left-padded batched forward
    pass; results are identical to scoring them one at a time.
    """
    import torch
    prompt_ids = tok(prompt, add_special_tokens=True).input_ids
    opt_ids = [tok(" " + o, add_special_tokens=False).input_ids
               for o in options]
    seqs = [prompt_ids + o for o in opt_ids]
    maxlen = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id or tok.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):  # left padding
        input_ids[i, maxlen - len(s):] = torch.tensor(s)
        attn[i, maxlen - len(s):] = 1
    keep = max(len(o) for o in opt_ids) + 1
    with torch.inference_mode():
        logits = model(input_ids.to(device),
                       attention_mask=attn.to(device),
                       logits_to_keep=keep).logits
    scores = []
    for i, o in enumerate(opt_ids):
        lp = torch.log_softmax(logits[i, :-1], dim=-1)
        n_opt = len(o)
        tgt = input_ids[i, -n_opt:]
        opt_lp = lp[-n_opt:].gather(1, tgt.unsqueeze(1)).sum().item()
        scores.append(opt_lp / n_opt)
    return scores


def run(model_idx, lang, max_items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, hf_id = MODELS[model_idx]
    items = load_items(lang)
    prog_path = os.path.join(PROG, f"m{model_idx}_{lang}.jsonl")
    done = set()
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [it for it in items if it["id"] not in done]
    print(f"=== {name} [{lang}] done={len(done)} todo={len(todo)}")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32)
    model.eval()

    with open(prog_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo[:max_items]):
            prompt = PROMPT[lang].format(passage=it["passage"],
                                         question=it["question"])
            scores = score_options(model, tok, prompt, it["options"])
            pred = int(max(range(4), key=lambda i: scores[i])) + 1
            rec = {"id": it["id"], "pred": pred,
                   "gold": it["correct_answer_num"],
                   "correct": pred == it["correct_answer_num"],
                   "scores": [round(s, 4) for s in scores]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (n + 1) % 10 == 0:
                print(f"  {len(done) + n + 1}/{len(items)}")
    print("window complete")


def merge():
    import csv
    rows = []
    detail = {}
    for mi, (name, hf_id) in enumerate(MODELS):
        row = {"model": name, "hf_id": hf_id}
        for lang in ["ne", "en"]:
            p = os.path.join(PROG, f"m{mi}_{lang}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                recs = [json.loads(line) for line in f]
            if not recs:
                continue
            acc = sum(r["correct"] for r in recs) / len(recs)
            row[f"acc_{lang}"] = round(acc, 4)
            row[f"n_{lang}"] = len(recs)
            detail[f"{name}_{lang}"] = recs
        if "acc_ne" in row and "acc_en" in row:
            row["drop_en_to_ne"] = round(row["acc_en"] - row["acc_ne"], 4)
        if len(row) > 2:
            rows.append(row)

    with open(os.path.join(RAW_OUT, "task2_qa.json"), "w",
              encoding="utf-8") as f:
        json.dump({"summaries": rows, "per_item": detail}, f, indent=1)
    if rows:
        fields = sorted({k for r in rows for k in r},
                        key=lambda k: (k != "model", k))
        with open(os.path.join(TAB_OUT, "task2_qa.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(r)
    print("merged -> results/raw/task2_qa.json, results/tables/task2_qa.csv")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        run(int(sys.argv[1]), sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9)
