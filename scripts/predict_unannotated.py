"""Run NER inference on unannotated rows and export rows needing annotation."""

import logging
from pathlib import Path

import pandas as pd
from transformers import pipeline

from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.inference.predictor import predict_batch, truncate_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def format_as_ner_tags(entity_list: list[dict], sentence: str) -> str:
    """Format entity predictions as NER_Tags tuples matching the CSV convention."""
    if not entity_list:
        return ""
    parts = []
    for e in entity_list:
        text = sentence[e["start"]:e["end"]]
        parts.append(f"({e['start']}, {e['end']}, '{text}', '{text.lower()}')")
    return "; ".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    checkpoint = root / "models" / "checkpoint-2000"
    csv_path = root / "data" / "annotated" / "260302_mentions_with_topics_clean.csv"
    export_path = root / "data" / "annotated" / "260316_to_annotate.csv"

    pretrained = "bioformers/bioformer-16L"
    tokenizer = get_tokenizer(pretrained)

    logger.info("Loading classifier from %s", checkpoint)
    classifier = pipeline(
        "ner",
        model=str(checkpoint),
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    df = pd.read_csv(csv_path)
    logger.info("Total rows: %d", len(df))

    # Rows where True? = False AND False? = False (unannotated)
    mask = (df["True?"] == False) & (df["False?"] == False)
    unannotated_idx = df.index[mask]
    logger.info("Unannotated rows to predict: %d", len(unannotated_idx))

    sentences = [
        truncate_if_needed(s, tokenizer, 512)
        for s in df.loc[unannotated_idx, "Sentence"]
    ]

    results = predict_batch(sentences, classifier, batch_size=16)

    # Store predictions in NER_Tags column
    ner_tags = [format_as_ner_tags(r, s) for r, s in zip(results, sentences)]
    df.loc[unannotated_idx, "NER_Tags"] = ner_tags

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    logger.info("Saved predictions to %s", csv_path)

    # Count how many predicted rows have empty NER_Tags
    predicted_empty = df.loc[unannotated_idx]
    predicted_empty = predicted_empty[predicted_empty["NER_Tags"].isna() | (predicted_empty["NER_Tags"] == "")]
    logger.info("Rows with empty NER_Tags (to annotate): %d", len(predicted_empty))

    # Export rows with empty NER_Tags to new file
    predicted_empty.to_csv(export_path, index=False)
    logger.info("Exported to %s", export_path)


if __name__ == "__main__":
    main()
