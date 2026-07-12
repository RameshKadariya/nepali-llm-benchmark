"""
Generate Task 2 and Task 3 figures from merged results.

Usage: python src/figures_task234.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# --- Task 2 paired QA ---
with open(os.path.join(RAW, "task2_paired_analysis.json")) as f:
    qa = json.load(f)

labels = [r["model"].replace(" (local, log-likelihood)", "\n(local)")
          for r in qa]
x = np.arange(len(qa))
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - 0.2, [r["acc_en"] * 100 for r in qa], width=0.38,
       color="#90a4ae", label="English")
ax.bar(x + 0.2, [r["acc_ne"] * 100 for r in qa], width=0.38,
       color="#c62828", label="Nepali (same questions)")
for i, r in enumerate(qa):
    ax.text(i - 0.2, r["acc_en"] * 100 + 1, f"{r['acc_en']*100:.1f}",
            ha="center", fontsize=8)
    ax.text(i + 0.2, r["acc_ne"] * 100 + 1, f"{r['acc_ne']*100:.1f}",
            ha="center", fontsize=8)
ax.axhline(25, color="black", ls="--", lw=1)
ax.text(len(qa) - 0.5, 26, "chance (25%)", fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Reading comprehension on identical questions: "
             "English vs Nepali (Belebele, paired items)")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task2_qa_drop.png"), dpi=200)
plt.close()

# --- Task 3 sentiment ---
with open(os.path.join(RAW, "task3_sentiment.json"), encoding="utf-8") as f:
    st = json.load(f)

labels = [f"{r['model']}\n(n={r['n']})" for r in st]
x = np.arange(len(st))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, [r["accuracy"] * 100 for r in st], width=0.38,
       color="#1565c0", label="Accuracy")
ax.bar(x + 0.2, [r["macro_f1"] * 100 for r in st], width=0.38,
       color="#6a1b9a", label="Macro F1")
for i, r in enumerate(st):
    ax.text(i - 0.2, r["accuracy"] * 100 + 1, f"{r['accuracy']*100:.1f}",
            ha="center", fontsize=8)
    ax.text(i + 0.2, r["macro_f1"] * 100 + 1, f"{r['macro_f1']*100:.1f}",
            ha="center", fontsize=8)
ax.axhline(100 / 3, color="black", ls="--", lw=1)
ax.text(len(st) - 0.5, 35, "chance (33.3%)", fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Score (%)")
ax.set_title("Sentiment classification of native Nepali social media text "
             "(3-class)")
ax.set_ylim(0, 80)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task3_sentiment.png"), dpi=200)
plt.close()

print("Wrote task2_qa_drop.png and task3_sentiment.png")
