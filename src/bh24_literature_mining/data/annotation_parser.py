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
        results = []
        for span in x.split("; "):
            span = span.strip()
            if not span:
                continue
            try:
                parsed = ast.literal_eval(span)
                if isinstance(parsed, tuple) and len(parsed) >= 4:
                    results.append(parsed)
            except (ValueError, SyntaxError):
                logger.warning("Failed to parse NER_Tags span: %s", span[:50])
        return results if results else None

    df["NER_Tags"] = df["NER_Tags"].apply(_safe_parse)

    positives = df[df["NER_Tags"].notna()].copy()
    negatives = df[df["NER_Tags"].isna()].copy()

    def _flatten_tags(tags_series: pd.Series) -> list:
        flat = []
        for tag_list in tags_series:
            if isinstance(tag_list, list):
                flat.extend(tag_list)
        return flat

    grouped_positives = (
        positives.groupby(["Sentence", "PMCID"])["NER_Tags"]
        .apply(_flatten_tags)
        .reset_index()
    )
    grouped_positives["NER_Tags"] = grouped_positives["NER_Tags"].apply(resolve_overlapping_spans)

    negatives_deduped = (
        negatives.drop_duplicates(subset=["Sentence", "PMCID"])[["PMCID", "Sentence"]]
        .copy()
    )
    negatives_deduped["NER_Tags"] = None

    return pd.concat([grouped_positives, negatives_deduped], ignore_index=True)


def resolve_overlapping_spans(tags: list) -> list:
    if not tags or len(tags) < 2:
        return tags
    sorted_tags = sorted(tags, key=lambda t: (t[1] - t[0]), reverse=True)
    kept = []
    for tag in sorted_tags:
        s, e = tag[0], tag[1]
        if not any(s < ke and e > ks for ks, ke, *_ in kept):
            kept.append(tag)
    return sorted(kept, key=lambda t: t[0])


def normalize_entity_type(df: pd.DataFrame, entity_type: str = "BT") -> pd.DataFrame:
    df = df.copy()
    df["NER_Tags"] = df["NER_Tags"].apply(
        lambda x: [[item[0], item[1], item[2], entity_type] for item in x] if x else None
    )
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_annotation_dataframe(
    df: pd.DataFrame,
    entity_type: str | None = "BT",
    include_negatives: bool = True,
) -> pd.DataFrame:
    df = filter_checked(df, include_negatives=include_negatives)
    df = parse_ner_tags(df)
    if entity_type is not None:
        df = normalize_entity_type(df, entity_type)
    return df


def prepare_annotations(
    paths: Path | list[Path],
    entity_type: str | None = "BT",
    include_negatives: bool = True,
) -> pd.DataFrame:
    return prepare_annotation_dataframe(
        load_annotations(paths),
        entity_type=entity_type,
        include_negatives=include_negatives,
    )
