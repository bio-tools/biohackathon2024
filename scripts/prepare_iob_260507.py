"""Prepare IOB splits from 260507_annotated.csv."""
from pathlib import Path

import pandas as pd

from bh24_literature_mining.data.annotation_parser import prepare_annotations
from bh24_literature_mining.data.cleaning import clean
from bh24_literature_mining.data.integrity import check_integrity_of_files
from bh24_literature_mining.data.iob_converter import convert_to_IOB_format_from_df
from bh24_literature_mining.data.splitter import split_by_pmcid
from bh24_literature_mining.data.tokenizer import get_tokenizer

CSV = Path("data/annotated/260507_annotated_fixed.csv")
CLEAN_CSV = Path("data/annotated/260507_annotated_clean.csv")
OUT = Path("data/IOB_260507")
MODEL = "bioformers/bioformer-16L"
SEED = 42

OUT.mkdir(parents=True, exist_ok=True)

clean(CSV).to_csv(CLEAN_CSV, index=False)
df = prepare_annotations(CLEAN_CSV, include_negatives=False)
tokenizer = get_tokenizer(MODEL)

train_df, val_df, test_df = split_by_pmcid(df, random_seed=SEED)
print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

convert_to_IOB_format_from_df(train_df, OUT, "train_IOB.tsv", tokenizer)
convert_to_IOB_format_from_df(val_df, OUT, "val_IOB.tsv", tokenizer)
convert_to_IOB_format_from_df(test_df, OUT, "test_IOB.tsv", tokenizer)
print(f"Saved IOB files to {OUT}")

check_integrity_of_files(
    [OUT / "train_IOB.tsv"], [OUT / "val_IOB.tsv"], [OUT / "test_IOB.tsv"]
)
