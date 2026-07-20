from pathlib import Path

import pandas as pd

from bh24_literature_mining.data.annotation_parser import (
    normalize_entity_type,
    prepare_annotation_dataframe,
)
from bh24_literature_mining.data.cleaning import clean
from bh24_literature_mining.data.integrity import check_integrity_of_files
from bh24_literature_mining.data.iob_converter import convert_to_IOB_format_from_df
from bh24_literature_mining.data.splitter import split_by_pmcid_and_resource
from bh24_literature_mining.data.tokenizer import get_tokenizer


def count_negative_rows(df: pd.DataFrame) -> int:
    return int(df["NER_Tags"].map(lambda tags: tags is None).sum())


def count_all_o_sentences(path: Path) -> int:
    count = 0
    has_tokens = False
    all_o = True
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if has_tokens and all_o:
                    count += 1
                has_tokens = False
                all_o = True
                continue
            has_tokens = True
            _, tag = stripped.split("\t", maxsplit=1)
            if tag != "O":
                all_o = False
    if has_tokens and all_o:
        count += 1
    return count


def prepare_iob_splits(
    annotations_path: Path,
    output_dir: Path,
    tokenizer_name: str,
    random_seed: int,
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_annotation_dataframe(
        clean(annotations_path),
        entity_type=None,
        include_negatives=True,
    )
    expected_negatives = count_negative_rows(df)
    if expected_negatives == 0:
        raise ValueError("No confirmed negative rows found in prepared annotations")

    train_df, validation_df, test_df = split_by_pmcid_and_resource(
        df,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        random_seed=random_seed,
    )
    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    negative_counts = {
        name: count_negative_rows(split) for name, split in splits.items()
    }
    if sum(negative_counts.values()) != expected_negatives:
        raise ValueError("Confirmed negative rows were lost during dataset splitting")
    if any(count == 0 for count in negative_counts.values()):
        raise ValueError(f"A dataset split contains no negative rows: {negative_counts}")

    tokenizer = get_tokenizer(tokenizer_name)
    filenames = {
        "train": "train_IOB.tsv",
        "validation": "val_IOB.tsv",
        "test": "test_IOB.tsv",
    }
    for name, split in splits.items():
        normalized = normalize_entity_type(split, "BT")
        convert_to_IOB_format_from_df(
            normalized,
            output_dir,
            filenames[name],
            tokenizer,
        )
        all_o_sentences = count_all_o_sentences(output_dir / filenames[name])
        if all_o_sentences != negative_counts[name]:
            raise ValueError(
                f"Expected {negative_counts[name]} all-O {name} sentences, "
                f"found {all_o_sentences}"
            )

    check_integrity_of_files(
        [output_dir / filenames["train"]],
        [output_dir / filenames["validation"]],
        [output_dir / filenames["test"]],
    )
    return {
        name: {
            "sentences": len(split),
            "negatives": negative_counts[name],
        }
        for name, split in splits.items()
    }
