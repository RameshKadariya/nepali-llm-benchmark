# Data Provenance

All evaluation data in this repository is derived from openly licensed
public datasets. This document records exactly where every frozen file
comes from and how it was constructed. The freezing script is
`src/prepare_data.py` (random seed 42). Frozen files are the single
source of truth for all experiments; live sources are never queried at
evaluation time.

## Source 1: Belebele

- Repository: https://huggingface.co/datasets/facebook/belebele
- License: CC-BY-SA 4.0
- Reference: Bandarkar et al., "The Belebele Benchmark: a Parallel
  Reading Comprehension Dataset in 122 Language Variants", ACL 2024.
- Subsets used: `npi_Deva` (Nepali) and `eng_Latn` (English), test
  split, 900 items each. Items are aligned across languages by the
  (`link`, `question_number`) key, so every Nepali question has an
  exact English counterpart over the same passage content. Passages
  originate from FLORES-200.

Derived frozen files:

| File | Contents | Size |
| --- | --- | --- |
| `frozen/qa_ne.jsonl` | Nepali multiple-choice reading comprehension | 200 |
| `frozen/qa_en.jsonl` | English counterparts of the same 200 items | 200 |
| `frozen/translation_pairs.jsonl` | Parallel ne-en FLORES passages | 150 |
| `frozen/retrieval_corpus_ne.jsonl` | Unique Nepali passages (corpus) | 488 |
| `frozen/retrieval_queries.jsonl` | Queries (ne and en) with gold passage ids | 200 |

Notes on the translation set: FLORES+ and IN22 are gated on the
HuggingFace Hub. Belebele redistributes the FLORES passages under
CC-BY-SA 4.0 with cross-language alignment, so the parallel passage
pairs used here are extracted from Belebele. Translation is therefore
evaluated at passage level (typically 2 to 5 sentences per passage).

Notes on the retrieval set: queries are Belebele questions and the gold
document is the passage the question was written against. One question
per unique passage is used to avoid corpus frequency bias.

## Source 2: NepaliSentiment

- Repository: https://huggingface.co/datasets/Shushant/NepaliSentiment
- Contents: native Nepali user-generated social media text (primarily
  YouTube and stock market forum comments), 6000 rows, labels 0/1/2.
- Label semantics are not documented in the dataset card. They were
  determined by manual inspection of samples per class and confirmed
  by native speaker validation (see `validation/`):
  0 = negative, 1 = positive, 2 = neutral.

Cleaning applied before sampling:

1. Dropped rows with malformed labels (8 rows: values such as `-`,
   `20`, `o`).
2. Dropped texts shorter than 15 or longer than 500 characters.
3. Required at least 10 Devanagari characters per text, removing
   English-only or emoji-only rows.
4. Exact-duplicate texts removed.

Resulting class pools: negative 1299, positive 1701, neutral 853.
A stratified sample of 70 per class (210 total) was frozen to
`frozen/sentiment_ne.jsonl`.

## Native speaker validation

`validation/validation_sample.jsonl` contains 30 randomly selected
items (15 sentiment, 10 QA, 5 translation) reviewed by a native
Nepali speaker (the author). The completed verdicts are stored in the
same directory and summarized in the paper's methodology section.

## Reproducing the frozen sets

```
pip install -r requirements.txt
python src/prepare_data.py
```

The script is deterministic given seed 42 and the upstream dataset
revisions current as of July 2026.
