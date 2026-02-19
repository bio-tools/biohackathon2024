#!/usr/bin/env python3
"""Repopulate IOB files from 260219_merge.csv using updated tokenizer."""

import logging
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.bh24_literature_mining.data.annotation_parser import (
    filter_checked,
    parse_ner_tags,
    normalize_entity_type,
)
from src.bh24_literature_mining.data.iob_converter import convert_to_IOB_format_from_df
from src.bh24_literature_mining.data.splitter import split_by_pmcid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    data_dir = Path(__file__).parent / "data"
    iob_dir = data_dir / "IOB"
    csv_path = data_dir / "annotated" / "260219_merge_cleaned.csv"

    tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-16L")

    # Load and filter checked rows
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    df = filter_checked(df)
    logger.info(f"After filtering checked rows: {len(df)} rows")

    # Parse NER_Tags and normalize entity type
    df = parse_ner_tags(df)
    df = normalize_entity_type(df, entity_type="BT")
    logger.info(f"After parsing NER_Tags: {len(df)} rows")

    # Split by PMCID (60% train, 40% test)
    train_df, val_df, test_df = split_by_pmcid(
        df, val_test_size=0.4, test_ratio_of_remainder=0.4, random_seed=42
    )

    logger.info(f"train split: {len(train_df)} rows")
    logger.info(f"Test split: {len(test_df)} rows")

    # Convert to IOB format and save
    convert_to_IOB_format_from_df(train_df, iob_dir, "train_IOB.tsv", tokenizer)
    convert_to_IOB_format_from_df(test_df, iob_dir, "test_IOB.tsv", tokenizer)
    convert_to_IOB_format_from_df(val_df, iob_dir, "val_IOB.tsv", tokenizer)

    logger.info("Saved train_IOB.tsv, val_IOB.tsv and test_IOB.tsv to %s", iob_dir)


if __name__ == "__main__":
    main()
