"""Recover missing entity annotations in merged_annotated.csv using NER model,
dictionary matching, and duplicate mention detection."""

import ast
import logging
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.inference.predictor import filter_predictions, predict_batch, truncate_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "sweep" / "checkpoint-2000"
INPUT = ROOT / "data" / "annotated" / "merged_annotated.csv"
BIOTOOLS = ROOT / "biotoolspub" / "biotoolspub_with_topic.tsv"
OUT_MULTIPLE = ROOT / "data" / "annotated" / "260316_to_annotate_multiple.csv"
OUT_AUTO = ROOT / "data" / "annotated" / "260316_auto_added.csv"
OUT_REVIEW = ROOT / "data" / "annotated" / "260316_to_review.csv"

HIGH_CONF = 0.8
LOW_CONF = 0.5


def parse_ner_tags(val: str) -> list[tuple]:
    if not val or (isinstance(val, float)):
        return []
    results = []
    for match in re.finditer(r"\((\d+),\s*(\d+),\s*'([^']*)',\s*'([^']*)'\)", str(val)):
        results.append((int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)))
    return results


def spans_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def find_all_occurrences(sentence: str, surface: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    lower_sent = sentence.lower()
    lower_surf = surface.lower()
    while True:
        idx = lower_sent.find(lower_surf, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(surface)))
        start = idx + 1
    return spans


def build_tool_vocab(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    names = df["name"].dropna().unique().tolist()
    return [n for n in names if isinstance(n, str) and len(n) >= 3]


def run_ner(sentences: list[str], tokenizer, checkpoint: Path) -> list[list[dict]]:
    classifier = pipeline(
        "ner",
        model=str(checkpoint),
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )
    return predict_batch(sentences, classifier, batch_size=16)


def merge_spans(existing: list[tuple], candidates: list[tuple]) -> list[tuple]:
    merged = list(existing)
    for cand in candidates:
        if not any(spans_overlap(cand, ex) for ex in merged):
            merged.append(cand)
    return sorted(merged, key=lambda x: x[0])


def format_ner_tags(spans: list[tuple]) -> str:
    return "; ".join(f"({s}, {e}, '{surf}', '{norm}')" for s, e, surf, norm in spans)


def main() -> None:
    df = pd.read_csv(INPUT)
    tokenizer = get_tokenizer("bioformers/bioformer-16L")

    logger.info("Running NER on %d sentences", len(df))
    sentences = [truncate_if_needed(s, tokenizer, 512) for s in df["Sentence"]]
    raw_predictions = run_ner(sentences, tokenizer, CHECKPOINT)

    logger.info("Building tool vocabulary")
    tool_vocab = build_tool_vocab(BIOTOOLS)
    tool_vocab_sorted = sorted(tool_vocab, key=len, reverse=True)

    auto_rows, review_rows, multiple_rows = [], [], []

    raw_predictions = [filter_predictions(p) for p in raw_predictions]

    for idx, (row, preds, sentence) in enumerate(zip(df.itertuples(), raw_predictions, sentences)):
        existing = parse_ner_tags(row.NER_Tags)
        existing_spans = [(s, e) for s, e, *_ in existing]

        # --- Model predictions ---
        new_high, new_mid = [], []
        for p in preds:
            span = (p["start"], p["end"])
            if any(spans_overlap(span, ex) for ex in existing_spans):
                continue
            surf = sentence[p["start"]:p["end"]]
            tag = (p["start"], p["end"], surf, surf.lower())
            if p["score"] >= HIGH_CONF:
                new_high.append(tag)
            elif p["score"] >= LOW_CONF:
                new_mid.append(tag)

        # --- Dictionary matching ---
        dict_candidates = []
        for tool in tool_vocab_sorted:
            for start, end in find_all_occurrences(sentence, tool):
                span = (start, end)
                if not any(spans_overlap(span, ex) for ex in existing_spans):
                    if not any(spans_overlap(span, (t[0], t[1])) for t in dict_candidates):
                        dict_candidates.append((start, end, sentence[start:end], tool.lower()))

        # --- Duplicate mention detection ---
        dup_candidates = []
        for s, e, surf, norm in existing:
            for start, end in find_all_occurrences(sentence, surf):
                if (start, end) != (s, e):
                    span = (start, end)
                    if not any(spans_overlap(span, ex) for ex in existing_spans):
                        dup_candidates.append((start, end, surf, norm))

        all_new = merge_spans([], new_high + dict_candidates + dup_candidates)
        review_spans = merge_spans([], new_mid)

        row_dict = {
            "PMCID": row.PMCID,
            "Sentence": row.Sentence,
            "True?": row._3,
            "False?": row._4,
            "NER_Tags": row.NER_Tags,
        }

        if len(preds) > 1:
            multiple_rows.append({**row_dict, "Model_Predictions": str(preds)})

        if all_new:
            merged = merge_spans(existing, all_new)
            auto_rows.append({**row_dict, "New_NER_Tags": format_ner_tags(merged), "Added_Spans": format_ner_tags(all_new)})

        if review_spans:
            review_rows.append({**row_dict, "Review_Spans": format_ner_tags(review_spans)})

    pd.DataFrame(multiple_rows).to_csv(OUT_MULTIPLE, index=False)
    pd.DataFrame(auto_rows).to_csv(OUT_AUTO, index=False)
    pd.DataFrame(review_rows).to_csv(OUT_REVIEW, index=False)

    logger.info("Multiple entities: %d rows → %s", len(multiple_rows), OUT_MULTIPLE)
    logger.info("Auto-added: %d rows → %s", len(auto_rows), OUT_AUTO)
    logger.info("To review: %d rows → %s", len(review_rows), OUT_REVIEW)


if __name__ == "__main__":
    main()
