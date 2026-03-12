import argparse
import logging
from pathlib import Path

import pandas as pd

from bh24_literature_mining.config import load_config
from bh24_literature_mining.data.annotation_parser import prepare_annotations
from bh24_literature_mining.data.augmentation import augment_dataframe, build_tool_vocab
from bh24_literature_mining.data.iob_converter import convert_to_IOB_format_from_df
from bh24_literature_mining.data.splitter import split_by_pmcid
from bh24_literature_mining.data.tokenizer import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IOB data from annotations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations-path", type=Path, default=None)
    parser.add_argument("--augment", action="store_true", default=None)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    args = parser.parse_args()

    project_root = Path(args.config).resolve().parent.parent
    config = load_config(args.config)

    annotations_path = args.annotations_path or (project_root / config.data.annotations_path)
    iob_dir = project_root / config.data.iob_dir
    iob_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading annotations from %s", annotations_path)
    df = prepare_annotations(annotations_path, include_negatives=config.data.include_negatives)

    logger.info("Splitting by PMCID")
    train_df, val_df, test_df = split_by_pmcid(
        df,
        val_test_size=config.data.val_ratio + config.data.test_ratio,
        test_ratio_of_remainder=config.data.test_ratio
        / (config.data.val_ratio + config.data.test_ratio),
        random_seed=config.data.random_seed,
    )

    do_augment = args.augment if args.augment is not None else config.data.augment
    if do_augment:
        biotools_path = project_root / config.data.biotools_path
        logger.info("Building tool vocabulary from %s", biotools_path)
        tool_vocab = build_tool_vocab(biotools_path)
        logger.info("Tool vocabulary size: %d", len(tool_vocab))

        logger.info("Augmenting training data with %d copies", config.data.augment_n_copies)
        aug_df = augment_dataframe(
            train_df, tool_vocab,
            n_copies=config.data.augment_n_copies,
            seed=config.data.random_seed,
        )
        logger.info("Augmented training set: %d original + %d augmented = %d total",
                     len(train_df), len(aug_df), len(train_df) + len(aug_df))
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=config.data.random_seed).reset_index(drop=True)

    tokenizer = get_tokenizer(config.model.pretrained)

    for split_df, filename in [
        (train_df, "train_IOB.tsv"),
        (val_df, "val_IOB.tsv"),
        (test_df, "test_IOB.tsv"),
    ]:
        logger.info("Writing %s (%d sentences)", filename, len(split_df))
        convert_to_IOB_format_from_df(split_df, iob_dir, filename, tokenizer)

    logger.info("Done. IOB files written to %s", iob_dir)


if __name__ == "__main__":
    main()
