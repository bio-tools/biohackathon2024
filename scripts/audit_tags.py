#!/usr/bin/env python3
"""Verify span offsets in NER_Tags match sentence text."""

import ast
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "annotated" / "260226_annotated.csv"


def parse_tag(raw: str) -> tuple | None:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) >= 4:
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def audit(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    issues: list[dict] = []

    for idx, row in df.iterrows():
        sentence = str(row["Sentence"])
        raw_tags = row.get("NER_Tags")
        if pd.isna(raw_tags):
            continue

        tag = parse_tag(str(raw_tags))
        if tag is None:
            issues.append({"row": idx, "type": "parse_error", "detail": str(raw_tags)[:80]})
            continue

        start, end, name = int(tag[0]), int(tag[1]), str(tag[2])

        if start < 0 or end < 0:
            issues.append({"row": idx, "type": "negative_offset", "detail": f"({start}, {end})"})
            continue

        if start >= end:
            issues.append({"row": idx, "type": "empty_span", "detail": f"start={start} >= end={end}"})
            continue

        if end > len(sentence):
            issues.append({
                "row": idx,
                "type": "out_of_bounds",
                "detail": f"end={end} > len={len(sentence)}",
            })
            continue

        extracted = sentence[start:end]
        if extracted.lower() != name.lower():
            issues.append({
                "row": idx,
                "type": "text_mismatch",
                "detail": f"expected='{name}' got='{extracted}'",
            })
            continue

        if start > 0 and sentence[start - 1].isalnum():
            issues.append({
                "row": idx,
                "type": "start_mid_word",
                "detail": f"char before start: '{sentence[start - 1]}'",
            })

        if end < len(sentence) and sentence[end].isalnum():
            issues.append({
                "row": idx,
                "type": "end_mid_word",
                "detail": f"char after end: '{sentence[end]}'",
            })

    return issues


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    logger.info("Auditing %s", csv_path)
    issues = audit(csv_path)

    if not issues:
        logger.info("No alignment issues found.")
        return

    logger.warning("Found %d issues:", len(issues))
    by_type: dict[str, int] = {}
    for issue in issues:
        by_type[issue["type"]] = by_type.get(issue["type"], 0) + 1
        logger.warning("  Row %d [%s]: %s", issue["row"], issue["type"], issue["detail"])

    logger.info("Summary:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", t, c)


if __name__ == "__main__":
    main()
