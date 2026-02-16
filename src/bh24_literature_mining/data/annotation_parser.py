import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_annotations(paths: Path | list[Path]) -> pd.DataFrame:
    if isinstance(paths, Path):
        paths = [paths]
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def filter_checked(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["False?"] == True, "NER_Tags"] = None
    return df[(df["True?"] == True) | (df["False?"] == True)].reset_index(drop=True)


def parse_ner_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["PMCID", "Sentence", "NER_Tags"]].copy()
    df["NER_Tags"] = df["NER_Tags"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    grouped = (
        df.groupby(["Sentence", "PMCID"])["NER_Tags"]
        .apply(lambda x: [i for i in x if i is not None])
        .reset_index()
    )
    return grouped


def normalize_entity_type(df: pd.DataFrame, entity_type: str = "BT") -> pd.DataFrame:
    df = df.copy()
    df["NER_Tags"] = df["NER_Tags"].apply(
        lambda x: [[item[0], item[1], item[2], entity_type] for item in x] if x else None
    )
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_annotations(
    paths: Path | list[Path], entity_type: str = "BT"
) -> pd.DataFrame:
    df = load_annotations(paths)
    df = filter_checked(df)
    df = parse_ner_tags(df)
    df = normalize_entity_type(df, entity_type)
    return df
