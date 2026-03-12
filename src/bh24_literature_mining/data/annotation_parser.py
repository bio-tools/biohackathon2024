import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_annotations(paths: Path | list[Path]) -> pd.DataFrame:
    if isinstance(paths, Path):
        paths = [paths]
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)



def filter_checked(df: pd.DataFrame, include_negatives: bool = True) -> pd.DataFrame:
    df = df.copy()
    if include_negatives:
        is_negative = df["False?"].eq(True) & df["True?"].eq(False)
        df.loc[is_negative, "NER_Tags"] = None
        return df[df["True?"].eq(True) | df["False?"].eq(True)].reset_index(drop=True)
    return df[df["True?"].eq(True)].reset_index(drop=True)


def parse_ner_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["PMCID", "Sentence", "NER_Tags"]].copy()

    def _safe_parse(x: object) -> object:
        if not isinstance(x, str):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse NER_Tags: %s", x[:50])
            return None

    df["NER_Tags"] = df["NER_Tags"].apply(_safe_parse)

    positives = df[df["NER_Tags"].notna()].copy()
    negatives = df[df["NER_Tags"].isna()].copy()

    grouped_positives = (
        positives.groupby(["Sentence", "PMCID"])["NER_Tags"]
        .apply(lambda x: [i for i in x if i is not None])
        .reset_index()
    )

    negatives_deduped = (
        negatives.drop_duplicates(subset=["Sentence", "PMCID"])[["PMCID", "Sentence"]]
        .copy()
    )
    negatives_deduped["NER_Tags"] = None

    return pd.concat([grouped_positives, negatives_deduped], ignore_index=True)


def normalize_entity_type(df: pd.DataFrame, entity_type: str = "BT") -> pd.DataFrame:
    df = df.copy()
    df["NER_Tags"] = df["NER_Tags"].apply(
        lambda x: [[item[0], item[1], item[2], entity_type] for item in x] if x else None
    )
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_annotations(
    paths: Path | list[Path], entity_type: str = "BT", include_negatives: bool = True
) -> pd.DataFrame:
    df = load_annotations(paths)
    df = filter_checked(df, include_negatives=include_negatives)
    df = parse_ner_tags(df)
    df = normalize_entity_type(df, entity_type)
    return df
