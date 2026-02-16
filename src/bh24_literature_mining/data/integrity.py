import logging
from pathlib import Path

import pandas as pd

from bh24_literature_mining.data.iob_converter import load_iob_file

logger = logging.getLogger(__name__)


def check_data_integrity(df: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []
    found_b_ent = False

    for i, entity in enumerate(df["tag"]):
        if entity.startswith("B-"):
            found_b_ent = True
        elif entity.startswith("I-"):
            if not found_b_ent:
                issues.append(f"Row: {i}, Hanging I-ent without preceding B-ent.")
            elif entity[2:] != df["tag"].iloc[i - 1][2:]:
                issues.append(f"Row: {i}, Mismatch between B-ent and I-ent.")
        if entity == "O" or entity == "" or (df["token"].iloc[i] == "" and entity == ""):
            found_b_ent = False

    return (True, []) if not issues else (False, issues)


def check_integrity_of_files(
    train_file_paths: list[Path],
    dev_file_paths: list[Path],
    test_file_paths: list[Path],
) -> None:
    for i, (train_file, dev_file, test_file) in enumerate(
        zip(train_file_paths, dev_file_paths, test_file_paths)
    ):
        logger.info("Checking Dataset %d", i + 1)
        for path in [train_file, dev_file, test_file]:
            temp_df = load_iob_file(path)
            is_valid, issues = check_data_integrity(temp_df)
            if is_valid:
                logger.info("%s is valid", path)
            else:
                logger.warning("%s has issues: %s", path, "; ".join(issues))
