"""Run NER inference on merged_annotated.csv using stilted-sweep-16 best checkpoint."""

import logging
from pathlib import Path

import pandas as pd
from transformers import pipeline

from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.inference.predictor import filter_predictions, predict_batch, truncate_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "annotated" / "merged_annotated.csv"
OUTPUT = ROOT / "data" / "annotated" / "merged_annotated_predicted.csv"
CHECKPOINT = ROOT / "models" / "sweep" / "stilted-sweep-16" / "checkpoint-1000"
PRETRAINED = "bioformers/bioformer-16L"


def format_ner_tags(preds: list[dict], sentence: str) -> str:
    if not preds:
        return ""
    return "; ".join(
        f"({p['start']}, {p['end']}, '{sentence[p['start']:p['end']]}', '{sentence[p['start']:p['end']].lower()}')"
        for p in preds
    )


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    logger.info("Using checkpoint: %s", CHECKPOINT)

    tokenizer = get_tokenizer(PRETRAINED)
    classifier = pipeline(
        "ner",
        model=str(CHECKPOINT),
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    df = pd.read_csv(INPUT)
    logger.info("Loaded %d sentences from %s", len(df), INPUT)

    sentences = [truncate_if_needed(s, tokenizer, 512) for s in df["Sentence"]]
    predictions = predict_batch(sentences, classifier, batch_size=16)

    df["NER_tags_predicted"] = [format_ner_tags(filter_predictions(p), s) for p, s in zip(predictions, sentences)]
    df.to_csv(OUTPUT, index=False)

    n_with_entities = (df["NER_tags_predicted"] != "").sum()
    logger.info("Predictions saved to %s  (%d/%d sentences with entities)", OUTPUT, n_with_entities, len(df))


if __name__ == "__main__":
    main()
