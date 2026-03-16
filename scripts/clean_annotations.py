#!/usr/bin/env python3
"""Clean annotation CSV: fix tag misalignment, remove oversized sentences,
and drop rows where the tag falls inside a URL."""

import ast
import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
LATEX_RE = re.compile(r"\\documentclass")
MAX_SENTENCE_LEN = 512


def parse_tag(raw: str) -> tuple | None:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) >= 4:
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def tag_overlaps_url(sentence: str, start: int, end: int) -> bool:
    for m in URL_RE.finditer(sentence):
        if start < m.end() and end > m.start():
            return True
    return False


def has_letter_boundary(sentence: str, start: int, end: int) -> bool:
    if start > 0 and sentence[start - 1].isalpha():
        return True
    if end < len(sentence) and sentence[end].isalpha():
        return True
    return False


def has_latex(sentence: str) -> bool:
    return bool(LATEX_RE.search(sentence))


def try_realign(sentence: str, name: str, start: int) -> tuple[int, int] | None:
    """Try to find the tool name as a standalone word in the sentence."""
    pattern = re.compile(r"(?<![a-zA-Z])" + re.escape(name) + r"(?![a-zA-Z])", re.IGNORECASE)
    matches = list(pattern.finditer(sentence))
    if len(matches) == 1:
        return matches[0].start(), matches[0].end()
    if matches:
        closest = min(matches, key=lambda m: abs(m.start() - start))
        return closest.start(), closest.end()
    return None


def clean(csv_path: Path, max_len: int = MAX_SENTENCE_LEN) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    n_orig = len(df)

    drop_idx: set = set()
    realigned = 0
    reasons: dict[str, int] = {}

    def mark_drop(idx: object, reason: str) -> None:
        drop_idx.add(idx)
        reasons[reason] = reasons.get(reason, 0) + 1

    for idx, row in df.iterrows():
        sentence = str(row["Sentence"])
        raw_tags = row.get("NER_Tags")

        if len(sentence) > max_len:
            mark_drop(idx, "sentence_too_long")
            continue

        if has_latex(sentence):
            mark_drop(idx, "latex_content")
            continue

        if pd.isna(raw_tags):
            continue

        tag = parse_tag(str(raw_tags))
        if tag is None:
            mark_drop(idx, "unparseable_tag")
            continue

        start, end, name = int(tag[0]), int(tag[1]), str(tag[2])

        if start < 0 or end < 0 or start >= end or end > len(sentence):
            mark_drop(idx, "invalid_span")
            continue

        if tag_overlaps_url(sentence, start, end):
            mark_drop(idx, "tag_in_url")
            continue

        if has_letter_boundary(sentence, start, end):
            new_span = try_realign(sentence, name, start)
            if new_span is not None:
                new_start, new_end = new_span
                if not has_letter_boundary(sentence, new_start, new_end):
                    extracted = sentence[new_start:new_end]
                    df.at[idx, "NER_Tags"] = str((new_start, new_end, extracted, tag[3]))
                    realigned += 1
                    continue
            mark_drop(idx, "tag_mid_word")

    df_clean = df.drop(index=list(drop_idx)).reset_index(drop=True)

    logger.info("Original rows: %d", n_orig)
    logger.info("Dropped rows: %d", len(drop_idx))
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", reason, count)
    logger.info("Realigned tags: %d", realigned)
    logger.info("Remaining rows: %d", len(df_clean))

    return df_clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean annotation CSV")
    parser.add_argument("input", type=Path, help="Input CSV path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--max-len", type=int, default=MAX_SENTENCE_LEN,
                        help=f"Max sentence length (default: {MAX_SENTENCE_LEN})")
    args = parser.parse_args()

    output = args.output or args.input.with_stem(args.input.stem + "_clean")

    df_clean = clean(args.input, max_len=args.max_len)
    df_clean.to_csv(output, index=False)
    logger.info("Saved to %s", output)


if __name__ == "__main__":
    main()
