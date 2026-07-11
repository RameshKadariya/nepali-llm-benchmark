"""
Generate Task 5 figure: monolingual vs cross-lingual retrieval quality
per embedding model (Recall@1 and Recall@5).

Usage: python src/figures_task5.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw", "task5_retrieval.json")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

with open(RAW, encoding="utf-8") as f:
    data = [r["summary"] for r in json.load(f)]
data.sort(key=lambda r: r["mono_recall@1"])

labels = [d["model"].replace(" (English-only control)", "\n(English-only control)")
          for d in data]
y = np.arange(len(data))

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(y + 0.2, [d["mono_recall@1"] for d in data], height=0.36,
        color="#1565c0", label="Nepali query -> Nepali corpus (R@1)")
ax.barh(y - 0.2, [d["cross_recall@1"] for d in data], height=0.36,
        color="#ef6c00", label="English query -> Nepali corpus (R@1)")
for i, d in enumerate(data):
    ax.text(d["mono_recall@1"] + 0.008, i + 0.2, f"{d['mono_recall@1']:.2f}",
            va="center", fontsize=8)
    ax.text(d["cross_recall@1"] + 0.008, i - 0.2, f"{d['cross_recall@1']:.2f}",
            va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Recall@1 over 488-passage Nepali corpus, 200 queries")
ax.set_title("Nepali retrieval quality by embedding model")
ax.set_xlim(0, 1.02)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task5_retrieval_r1.png"), dpi=200)
plt.close()
print("Wrote results/figures/task5_retrieval_r1.png")
