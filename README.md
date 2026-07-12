# How Well Does AI Understand Nepali?

A multi-task benchmark of large language models and text embedding
models on Nepali (नेपाली), a low-resource language written in the
Devanagari script and spoken by more than 30 million people.

**Author:** [Ramesh Kadariya](https://rameshkadariya.com.np) |
ISMT College (University of Sunderland), Chitwan, Nepal |
contact@rameshkadariya.com.np

Paper: `paper/` (PDF). Published on Zenodo and ResearchGate.

## Research question

Modern AI models are trained overwhelmingly on English. This project
measures, with fully reproducible experiments, how much capability
they retain on Nepali and where exactly they fail.

## Tasks

| # | Task | Data | Metric |
| --- | --- | --- | --- |
| 1 | Tokenizer fertility (ne vs en cost) | Parallel FLORES passages | tokens/word, ne:en ratio |
| 2 | Reading comprehension (paired ne/en) | Belebele | accuracy |
| 3 | Sentiment classification (native text) | NepaliSentiment | accuracy, macro-F1 |
| 4 | Machine translation (ne to en, en to ne) | Parallel FLORES passages | chrF++, BLEU |
| 5 | Embedding retrieval (mono + cross-lingual) | Belebele-derived corpus | Recall@k, MRR |

## Results

Headline findings (full details in the paper and results/):

**Task 1, tokenizer cost.** For identical parallel content, Nepali costs
4.79x more tokens than English on the GPT-4 (cl100k) tokenizer, 4.46x on
Qwen 2.5, 2.60x on Llama 3.1, but only 1.62x on GPT-4o and 1.19x on
NLLB-200. Tokenizer design, not the Devanagari script, drives the cost.

**Task 2, reading comprehension (paired, same questions).**

| Model | n | Acc EN | Acc NE | Drop | McNemar p |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B (local) | 37 | 46.0 | 29.7 | 16.2 | 0.18 |
| Llama 3.1 8B | 200 | 84.0 | 40.5 | 43.5 | 2.7e-23 |
| Llama 4 Scout 17B | 200 | 96.0 | 84.5 | 11.5 | 5.7e-06 |
| Llama 3.3 70B | 178 | 97.2 | 84.3 | 12.9 | 2.4e-07 |
| GPT-OSS 120B | 200 | 95.0 | 86.5 | 8.5 | 4.9e-04 |
| Gemini 3.1 Flash Lite | 200 | 97.0 | 89.5 | 7.5 | 6.1e-05 |

**Task 3, sentiment on native informal Nepali (chance 33.3%).** Best is
GPT-OSS 120B at 67.1% accuracy (n=158); Llama 8B reaches only 51.9%.
Informal register is far harder than formal Belebele text.

**Task 4, translation (chrF++).** Models read Nepali better than they
write it: Llama 8B scores 53.9 (ne to en) vs 30.6 (en to ne); Scout 17B
65.5 vs 49.1.

**Task 5, retrieval over a Nepali corpus (Recall@1).** BGE-M3 84.0%,
multilingual-e5-small 77.5%, LaBSE 58.0%, paraphrase-multilingual-MiniLM
15.0%, English-only all-MiniLM-L6 1.0%. Nepali RAG works today if the
embedding model is chosen carefully.

## Repository layout

```
data/
  PROVENANCE.md        exact origin of every data file
  frozen/              frozen evaluation sets (seed 42), JSONL
  validation/          native speaker validation samples
src/                   one runner script per task
results/
  raw/                 raw per-item model outputs (JSON)
  figures/             generated charts
  tables/              generated result tables
paper/                 manuscript
```

## Reproducing

```
pip install -r requirements.txt
python src/prepare_data.py        # rebuild frozen sets
python src/task1_tokenizer.py     # tokenizer fertility study
# further task scripts documented as they land
```

## License

Code: MIT. Frozen data derived from Belebele remains CC-BY-SA 4.0
(see `data/PROVENANCE.md`). Please cite via `CITATION.cff`.
