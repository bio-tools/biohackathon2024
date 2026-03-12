#!/usr/bin/env python3
"""Clean 260302_mentions_with_topics.csv and validate NER tag alignment."""

import ast
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "annotated" / "260302_mentions_with_topics.csv"
SENTENCE_LENGTH_THRESHOLD = 1000


def strip_non_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def span_in_url(sentence: str, start: int, end: int) -> bool:
    for m in re.finditer(r"https?://\S+", sentence):
        if start >= m.start() and end <= m.end():
            return True
    return False


def parse_tag(raw: str) -> tuple | None:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) >= 3:
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def fix_tag_offsets(sentence: str, tag: tuple) -> tuple | None:
    """Attempt to relocate span by searching for name in sentence."""
    name = str(tag[2])
    match = re.search(re.escape(name), sentence, re.IGNORECASE)
    if match:
        start, end = match.start(), match.end()
        return (start, end) + tag[2:]
    return None


def clean(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    original_len = len(df)
    logger.info("Loaded %d rows from %s", original_len, csv_path)

    df["Sentence"] = df["Sentence"].astype(str).apply(strip_non_ascii)
    df["NER_Tags"] = df["NER_Tags"].astype(str).apply(strip_non_ascii)

    mask_url: list[bool] = []
    for _, row in df.iterrows():
        tag = parse_tag(str(row["NER_Tags"]))
        if tag is None:
            mask_url.append(False)
            continue
        mask_url.append(span_in_url(str(row["Sentence"]), int(tag[0]), int(tag[1])))

    before = len(df)
    df = df[~pd.Series(mask_url, index=df.index)]
    logger.info("Removed %d rows with span inside URL", before - len(df))

    before = len(df)
    df = df[df["Sentence"].str.len() <= SENTENCE_LENGTH_THRESHOLD]
    logger.info(
        "Removed %d rows exceeding length threshold (%d)",
        before - len(df),
        SENTENCE_LENGTH_THRESHOLD,
    )

    fixed = 0
    dropped = 0
    rows_to_drop: list[int] = []

    for idx, row in df.iterrows():
        sentence = str(row["Sentence"])
        raw = str(row["NER_Tags"])
        tag = parse_tag(raw)
        if tag is None:
            rows_to_drop.append(idx)
            dropped += 1
            continue

        start, end, name = int(tag[0]), int(tag[1]), str(tag[2])

        if (
            start < 0
            or end < 0
            or start >= end
            or end > len(sentence)
            or sentence[start:end].lower() != name.lower()
        ):
            fixed_tag = fix_tag_offsets(sentence, tag)
            if fixed_tag is None:
                rows_to_drop.append(idx)
                dropped += 1
            else:
                df.at[idx, "NER_Tags"] = str(fixed_tag)
                fixed += 1

    df = df.drop(index=rows_to_drop)
    logger.info("Fixed %d misaligned tags; dropped %d unfixable rows", fixed, dropped)

    for col in ("True?", "False?"):
        if col not in df.columns:
            df[col] = False

    logger.info("Final row count: %d (removed %d total)", len(df), original_len - len(df))

    df.to_csv(out_path, index=False)
    logger.info("Saved cleaned data to %s", out_path)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_stem(csv_path.stem + "_clean")
    clean(csv_path, out_path)


if __name__ == "__main__":
    main()
