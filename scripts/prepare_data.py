#!/usr/bin/env python3
"""
Download, clean, filter, deduplicate, and tokenize the data mix for Erebus v2.

Mix:
  - FineWeb-Edu (score >= 4): ~5B tokens
  - Cosmopedia v2:            ~4B tokens
  - Python-Edu (smollm):      ~1B tokens
  Total:                      ~10B tokens

Quality filters (Gopher-inspired):
  - Document length bounds (100 chars min, 100K max)
  - Line-level repetition detection
  - N-gram repetition detection
  - Special character ratio
  - Mean word length bounds
  - Exact deduplication via SHA-256
  - Unicode normalization and whitespace cleanup
  - Boilerplate line stripping

Saves tokenized data as memory-mapped uint16 numpy arrays for fast disk-based training.
"""

import argparse
import hashlib
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "fineweb-edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "target_tokens": 5_000_000_000,
        "filter_fn": lambda x: x.get("score", 0) >= 4,
    },
    {
        "name": "cosmopedia-v2",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "subset": "cosmopedia-v2",
        "split": "train",
        "text_field": "text",
        "target_tokens": 4_000_000_000,
        "filter_fn": None,
    },
    {
        "name": "python-edu",
        "dataset": "jon-tow/starcoderdata-python-edu",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "target_tokens": 1_000_000_000,
        "filter_fn": lambda x: x.get("int_score", 0) >= 3,
        "is_code": True,
    },
    {
        "name": "fineweb-edu-3",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "default",
        "split": "train",
        "text_field": "text",
        "target_tokens": 4_700_000_000,
        "filter_fn": lambda x: x.get("score", 0) == 3,
    },
]


# ---------------------------------------------------------------------------
# Boilerplate patterns to strip (matched as full lines, case-insensitive)
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^\s*cookie\s*(policy|settings|preferences|consent).*$",
        r"^\s*terms\s*(of\s+service|and\s+conditions|of\s+use).*$",
        r"^\s*privacy\s*policy.*$",
        r"^\s*subscribe\s+to\s+(our\s+)?newsletter.*$",
        r"^\s*all\s+rights\s+reserved\.?\s*$",
        r"^\s*copyright\s*©?\s*\d{4}.*$",
        r"^\s*share\s+(this|on)\s+(facebook|twitter|linkedin|email).*$",
        r"^\s*follow\s+us\s+on\s+.*$",
        r"^\s*click\s+here\s+to\s+.*$",
        r"^\s*sign\s+up\s+(for|to)\s+.*$",
        r"^\s*leave\s+a\s+(comment|reply).*$",
        r"^\s*related\s+(articles?|posts?|stories)\s*:?\s*$",
        r"^\s*advertisement\s*$",
        r"^\s*sponsored\s*(content|post)?\s*$",
    ]
]


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------
class QualityFilter:
    """Light-touch Gopher-inspired quality heuristics."""

    def __init__(self, min_chars=100, max_chars=100_000,
                 max_line_dup_ratio=0.3, max_ngram_repeats=5, ngram_size=10,
                 max_special_ratio=0.20, min_mean_word_len=3, max_mean_word_len=12):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_line_dup_ratio = max_line_dup_ratio
        self.max_ngram_repeats = max_ngram_repeats
        self.ngram_size = ngram_size
        self.max_special_ratio = max_special_ratio
        self.min_mean_word_len = min_mean_word_len
        self.max_mean_word_len = max_mean_word_len

        self.stats = Counter()

    def check(self, text):
        self.stats["total"] += 1

        if len(text) < self.min_chars:
            self.stats["too_short"] += 1
            return False

        if len(text) > self.max_chars:
            self.stats["too_long"] += 1
            return False

        lines = text.split("\n")
        non_empty = [l.strip() for l in lines if l.strip()]
        if len(non_empty) > 1:
            line_counts = Counter(non_empty)
            dup_lines = sum(c - 1 for c in line_counts.values() if c > 1)
            if dup_lines / len(non_empty) > self.max_line_dup_ratio:
                self.stats["line_repetition"] += 1
                return False

        words = text.split()
        if len(words) >= self.ngram_size:
            ngrams = [" ".join(words[i:i+self.ngram_size])
                      for i in range(len(words) - self.ngram_size + 1)]
            ngram_counts = Counter(ngrams)
            if ngram_counts.most_common(1)[0][1] > self.max_ngram_repeats:
                self.stats["ngram_repetition"] += 1
                return False

        alnum_ws = sum(1 for c in text if c.isalnum() or c.isspace())
        if len(text) > 0 and (len(text) - alnum_ws) / len(text) > self.max_special_ratio:
            self.stats["special_chars"] += 1
            return False

        if words:
            mean_wl = sum(len(w) for w in words) / len(words)
            if mean_wl < self.min_mean_word_len or mean_wl > self.max_mean_word_len:
                self.stats["word_length"] += 1
                return False

        self.stats["passed"] += 1
        return True

    def report(self, source_name):
        total = self.stats["total"]
        if total == 0:
            return
        passed = self.stats["passed"]
        print(f"\n  [{source_name}] Quality filter stats:")
        print(f"    Total docs seen:    {total:>12,}")
        print(f"    Passed:             {passed:>12,} ({passed/total*100:.1f}%)")
        for key in ["too_short", "too_long", "line_repetition",
                     "ngram_repetition", "special_chars", "word_length"]:
            count = self.stats.get(key, 0)
            if count > 0:
                print(f"    Filtered ({key:>16s}): {count:>12,} ({count/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# Content cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(p.match(line) for p in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Tokenize and save one source
# ---------------------------------------------------------------------------
def _flush_buffer(f, token_buffer, buf_pos, total_tokens, target,
                   name, start_time, dedup_hits):
    """Write buffer to disk and print progress."""
    f.write(token_buffer[:buf_pos].tobytes())
    total_tokens += buf_pos
    elapsed = time.time() - start_time
    tps = total_tokens / elapsed
    pct = total_tokens / target * 100
    print(
        f"[{name}] {total_tokens:,} / {target:,} tokens "
        f"({pct:.1f}%) | {tps:,.0f} tok/s | "
        f"dedup_skipped={dedup_hits:,}",
        flush=True,
    )
    return total_tokens


def tokenize_and_save(source, tokenizer, output_dir, seen_hashes,
                      buffer_size=50_000_000, target_scale=1.0,
                      batch_size=64):
    name = source["name"]
    target = int(source["target_tokens"] * target_scale)
    out_path = output_dir / f"{name}.bin"
    meta_path = output_dir / f"{name}.meta"

    if meta_path.exists():
        with open(meta_path) as f:
            existing_tokens = int(f.read().strip())
        if existing_tokens >= target:
            print(f"[{name}] Already complete: {existing_tokens:,} tokens", flush=True)
            return existing_tokens

    print(f"\n[{name}] Loading dataset: {source['dataset']} / {source['subset']}", flush=True)
    print(f"[{name}] Target: {target:,} tokens", flush=True)

    load_kwargs = dict(
        path=source["dataset"],
        split=source["split"],
        streaming=True,
        trust_remote_code=True,
    )
    if source["subset"]:
        load_kwargs["name"] = source["subset"]
    ds = load_dataset(**load_kwargs)

    if source.get("is_code"):
        qf = QualityFilter(max_special_ratio=0.35, max_mean_word_len=20)
    else:
        qf = QualityFilter()
    eos_id = tokenizer.eos_token_id
    token_buffer = np.empty(buffer_size, dtype=np.uint32)
    buf_pos = 0
    total_tokens = 0
    dedup_hits = 0
    start_time = time.time()
    text_batch = []

    with open(out_path, "wb") as f:
        for i, sample in enumerate(ds):
            if source["filter_fn"] and not source["filter_fn"](sample):
                continue

            text = sample.get(source["text_field"], "")
            if not text or not text.strip():
                continue

            # Dedup on raw text BEFORE expensive cleaning
            raw_normalized = " ".join(text.split())
            doc_hash = hashlib.sha256(
                raw_normalized.encode("utf-8")
            ).digest()[:16]
            if doc_hash in seen_hashes:
                dedup_hits += 1
                continue
            seen_hashes.add(doc_hash)

            text = clean_text(text)
            if not text:
                continue

            if not qf.check(text):
                continue

            text_batch.append(text)

            if len(text_batch) >= batch_size or total_tokens + buf_pos >= target:
                encoded = tokenizer(text_batch, add_special_tokens=False)["input_ids"]
                for token_ids in encoded:
                    token_ids.append(eos_id)
                    tok_arr = np.array(token_ids, dtype=np.uint32)
                    n = len(tok_arr)

                    while n > 0:
                        space = buffer_size - buf_pos
                        chunk = min(n, space)
                        token_buffer[buf_pos:buf_pos + chunk] = tok_arr[:chunk]
                        buf_pos += chunk
                        tok_arr = tok_arr[chunk:]
                        n -= chunk

                        if buf_pos >= buffer_size:
                            total_tokens = _flush_buffer(
                                f, token_buffer, buf_pos, total_tokens,
                                target, name, start_time, dedup_hits)
                            buf_pos = 0

                text_batch = []

            if total_tokens + buf_pos >= target:
                break

        # Flush remaining text batch
        if text_batch:
            encoded = tokenizer(text_batch, add_special_tokens=False)["input_ids"]
            for token_ids in encoded:
                token_ids.append(eos_id)
                tok_arr = np.array(token_ids, dtype=np.uint32)
                n = len(tok_arr)

                while n > 0:
                    space = buffer_size - buf_pos
                    chunk = min(n, space)
                    token_buffer[buf_pos:buf_pos + chunk] = tok_arr[:chunk]
                    buf_pos += chunk
                    tok_arr = tok_arr[chunk:]
                    n -= chunk

                    if buf_pos >= buffer_size:
                        total_tokens = _flush_buffer(
                            f, token_buffer, buf_pos, total_tokens,
                            target, name, start_time, dedup_hits)
                        buf_pos = 0

        if buf_pos > 0:
            f.write(token_buffer[:buf_pos].tobytes())
            total_tokens += buf_pos

    with open(meta_path, "w") as f:
        f.write(str(total_tokens))

    elapsed = time.time() - start_time
    print(
        f"\n[{name}] Done: {total_tokens:,} tokens in {elapsed/3600:.1f}h "
        f"({total_tokens/elapsed:,.0f} tok/s) -> {out_path}",
        flush=True,
    )
    print(f"[{name}] Dedup skipped: {dedup_hits:,}", flush=True)
    qf.report(name)
    return total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare Erebus v2 training data (with quality filters)")
    parser.add_argument("--output_dir", type=str, default="/data/tokenized")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target_scale", type=float, default=1.0,
                        help="Scale all token targets (e.g., 0.001 for 10M token test run)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Erebus v2 — Data Preparation with Quality Filters")
    print("=" * 60)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)

    if args.target_scale != 1.0:
        print(f"Target scale: {args.target_scale} (test mode)", flush=True)

    seen_hashes = set()
    grand_total = 0

    for source in SOURCES:
        n = tokenize_and_save(source, tokenizer, output_dir, seen_hashes,
                              target_scale=args.target_scale)
        grand_total += n

    manifest = output_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for source in SOURCES:
            meta_path = output_dir / f"{source['name']}.meta"
            tokens = int(meta_path.read_text().strip())
            f.write(f"{source['name']}.bin\t{tokens}\n")

    print(f"\n{'=' * 60}")
    print(f"All done! Total: {grand_total:,} tokens")
    print(f"Unique docs hashed: {len(seen_hashes):,}")
    print(f"Manifest: {manifest}")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
