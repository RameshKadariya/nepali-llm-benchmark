"""
Task 1: Tokenizer fertility on Nepali vs English.

For each tokenizer, encode the 150 frozen parallel passages in both
languages and measure:

  - fertility: tokens per whitespace-delimited word
  - tokens per Unicode character
  - cost ratio: Nepali tokens / English tokens for the same content

The cost ratio is the headline number: it is exactly the factor by
which API costs and effective context capacity change when the same
information is expressed in Nepali instead of English.

Outputs:
  results/raw/task1_tokenizer.json    per-tokenizer, per-passage stats
  results/tables/task1_tokenizer.csv  summary table

Usage: python src/task1_tokenizer.py
"""

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data", "frozen")
RAW_OUT = os.path.join(ROOT, "results", "raw")
TAB_OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(RAW_OUT, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)

# (display name, kind, identifier, notes)
TOKENIZERS = [
    ("GPT-4o / o200k_base", "tiktoken", "o200k_base",
     "OpenAI GPT-4o family"),
    ("GPT-4 / cl100k_base", "tiktoken", "cl100k_base",
     "OpenAI GPT-4 and GPT-3.5 family"),
    ("GPT-2 / r50k", "tiktoken", "gpt2",
     "Legacy English-centric baseline"),
    ("Llama 3.1", "hf", "NousResearch/Meta-Llama-3.1-8B",
     "Meta Llama 3 tokenizer (ungated mirror)"),
    ("Qwen 2.5", "hf", "Qwen/Qwen2.5-7B-Instruct",
     "Alibaba Qwen family"),
    ("Mistral v0.3", "hf", "mistralai/Mistral-7B-v0.3",
     "Mistral family"),
    ("DeepSeek V3", "hf", "deepseek-ai/DeepSeek-V3",
     "DeepSeek family"),
    ("Gemma 2", "hf", "unsloth/gemma-2-9b",
     "Google Gemma tokenizer (ungated mirror)"),
    ("mT5", "hf", "google/mt5-base",
     "Massively multilingual encoder-decoder"),
    ("NLLB-200", "hf", "facebook/nllb-200-distilled-600M",
     "Translation-specialized, 200 languages"),
    ("BLOOM", "hf", "bigscience/bloom-560m",
     "Multilingual open model"),
    ("NepBERTa", "hf", "NepBERTa/NepBERTa",
     "Nepali-specific BERT"),
]


def load_pairs():
    path = os.path.join(FROZEN, "translation_pairs.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_encoder(kind, ident):
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(ident)
        return lambda s: len(enc.encode(s))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ident, use_fast=True,
                                        trust_remote_code=True)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} parallel passages.")
    all_results = []

    for name, kind, ident, notes in TOKENIZERS:
        try:
            count = get_encoder(kind, ident)
        except Exception as e:
            print(f"SKIP {name}: {str(e)[:100]}")
            continue

        per_passage = []
        for p in pairs:
            ne, en = p["ne"], p["en"]
            t_ne, t_en = count(ne), count(en)
            per_passage.append({
                "id": p["id"],
                "tokens_ne": t_ne,
                "tokens_en": t_en,
                "words_ne": len(ne.split()),
                "words_en": len(en.split()),
                "chars_ne": len(ne),
                "chars_en": len(en),
            })

        fert_ne = [r["tokens_ne"] / r["words_ne"] for r in per_passage]
        fert_en = [r["tokens_en"] / r["words_en"] for r in per_passage]
        ratio = [r["tokens_ne"] / r["tokens_en"] for r in per_passage]

        summary = {
            "tokenizer": name,
            "identifier": ident,
            "notes": notes,
            "fertility_ne_mean": round(statistics.mean(fert_ne), 3),
            "fertility_en_mean": round(statistics.mean(fert_en), 3),
            "cost_ratio_mean": round(statistics.mean(ratio), 3),
            "cost_ratio_median": round(statistics.median(ratio), 3),
            "cost_ratio_stdev": round(statistics.stdev(ratio), 3),
            "total_tokens_ne": sum(r["tokens_ne"] for r in per_passage),
            "total_tokens_en": sum(r["tokens_en"] for r in per_passage),
        }
        all_results.append({"summary": summary, "per_passage": per_passage})
        print(f"{name:24s} fert_ne={summary['fertility_ne_mean']:.2f} "
              f"fert_en={summary['fertility_en_mean']:.2f} "
              f"ratio={summary['cost_ratio_mean']:.2f}")

    with open(os.path.join(RAW_OUT, "task1_tokenizer.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    import csv
    with open(os.path.join(TAB_OUT, "task1_tokenizer.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(
            all_results[0]["summary"].keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(r["summary"])
    print("Wrote results/raw/task1_tokenizer.json and "
          "results/tables/task1_tokenizer.csv")


if __name__ == "__main__":
    main()
