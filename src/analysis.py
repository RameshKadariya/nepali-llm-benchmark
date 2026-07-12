"""
Consolidated analysis for the paper.

Reading comprehension (Task 2): for every model, accuracy is computed
on the paired subset of items completed in BOTH languages, so the
English-to-Nepali comparison is strictly within-model on identical
content. Significance of the language effect is tested with McNemar's
exact test on the paired correctness table.

Also emits the final consolidated tables used in the paper.

Usage: python src/analysis.py
"""

import json
import os

from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
TAB = os.path.join(ROOT, "results", "tables")


def read_prog(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r.get("id")}


def mcnemar_exact(b, c):
    """Exact McNemar test on discordant pairs (b: en right ne wrong,
    c: en wrong ne right)."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5).pvalue * 1.0


QA_RUNS = [
    ("Qwen2.5-0.5B (local, log-likelihood)",
     os.path.join(RAW, "task2_progress", "m0_{lang}.jsonl")),
    ("Llama 3.1 8B", os.path.join(RAW, "task2_progress",
                                  "api_llama8b_{lang}.jsonl")),
    ("Llama 4 Scout 17B", os.path.join(RAW, "task2_progress",
                                       "api_scout17b_{lang}.jsonl")),
    ("Llama 3.3 70B", os.path.join(RAW, "task2_progress",
                                   "api_llama70b_{lang}.jsonl")),
    ("GPT-OSS 120B", os.path.join(RAW, "task2_progress",
                                  "api_gptoss120b_{lang}.jsonl")),
    ("Gemini 3.1 Flash Lite", os.path.join(RAW, "task2_progress",
                                           "api_geminilite_{lang}.jsonl")),
]


def main():
    rows = []
    for name, tmpl in QA_RUNS:
        en = read_prog(tmpl.format(lang="en"))
        ne = read_prog(tmpl.format(lang="ne"))
        common = sorted(set(en) & set(ne))
        if not common:
            continue
        acc_en = sum(en[i]["correct"] for i in common) / len(common)
        acc_ne = sum(ne[i]["correct"] for i in common) / len(common)
        b = sum(1 for i in common
                if en[i]["correct"] and not ne[i]["correct"])
        c = sum(1 for i in common
                if not en[i]["correct"] and ne[i]["correct"])
        p = mcnemar_exact(b, c)
        rows.append({
            "model": name, "paired_n": len(common),
            "acc_en": round(acc_en, 4), "acc_ne": round(acc_ne, 4),
            "drop": round(acc_en - acc_ne, 4),
            "discordant_en_only": b, "discordant_ne_only": c,
            "mcnemar_p": float(f"{p:.2e}"),
        })

    with open(os.path.join(RAW, "task2_paired_analysis.json"), "w") as f:
        json.dump(rows, f, indent=1)
    import csv
    with open(os.path.join(TAB, "task2_paired.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'model':38s} {'n':>4} {'en':>6} {'ne':>6} {'drop':>6} "
          f"{'p':>9}")
    for r in rows:
        print(f"{r['model']:38s} {r['paired_n']:>4} {r['acc_en']:>6.3f} "
              f"{r['acc_ne']:>6.3f} {r['drop']:>6.3f} "
              f"{r['mcnemar_p']:>9.2e}")


if __name__ == "__main__":
    main()
