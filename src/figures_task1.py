"""
Generate Task 1 figures from results/raw/task1_tokenizer.json.

Figure 1: Nepali-to-English token cost ratio per tokenizer.
Figure 2: Fertility (tokens per word) in Nepali vs English.

Usage: python src/figures_task1.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw", "task1_tokenizer.json")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

with open(RAW, encoding="utf-8") as f:
    data = [r["summary"] for r in json.load(f)]

data.sort(key=lambda r: r["cost_ratio_mean"])

names = [r["tokenizer"] for r in data]
ratios = [r["cost_ratio_mean"] for r in data]
colors = ["#2e7d32" if v <= 1.5 else "#f9a825" if v <= 3 else "#c62828"
          for v in ratios]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(names, ratios, color=colors)
ax.axvline(1.0, color="black", lw=1, ls="--")
ax.text(1.02, -0.7, "parity", fontsize=8)
for b, v in zip(bars, ratios):
    ax.text(v + 0.06, b.get_y() + b.get_height() / 2, f"{v:.2f}x",
            va="center", fontsize=9)
ax.set_xlabel("Token cost ratio: Nepali / English (same content, "
              "150 parallel FLORES passages)")
ax.set_title("How much more expensive is Nepali than English, "
             "per tokenizer?")
ax.set_xlim(0, max(ratios) + 0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task1_cost_ratio.png"), dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5.5))
d2 = sorted(data, key=lambda r: r["fertility_ne_mean"])
y = range(len(d2))
ax.barh([i + 0.2 for i in y], [r["fertility_ne_mean"] for r in d2],
        height=0.4, label="Nepali", color="#1565c0")
ax.barh([i - 0.2 for i in y], [r["fertility_en_mean"] for r in d2],
        height=0.4, label="English", color="#90a4ae")
ax.set_yticks(list(y))
ax.set_yticklabels([r["tokenizer"] for r in d2])
ax.set_xlabel("Fertility (tokens per whitespace word)")
ax.set_title("Tokenizer fertility: Nepali vs English")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task1_fertility.png"), dpi=200)
plt.close()

print("Wrote figures to results/figures/")
