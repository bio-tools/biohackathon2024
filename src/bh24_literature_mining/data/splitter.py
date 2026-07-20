import random

import pandas as pd
from sklearn.model_selection import train_test_split


def get_resource_ids(tags: object) -> set[str]:
    if not isinstance(tags, list):
        return set()
    return {
        str(tag[3])
        for tag in tags
        if isinstance(tag, (list, tuple)) and len(tag) >= 4
    }


def get_document_tool_components(df: pd.DataFrame) -> list[set[str]]:
    pmcids = sorted(str(pmcid) for pmcid in df["PMCID"].unique())
    parent = {pmcid: pmcid for pmcid in pmcids}

    def find(pmcid: str) -> str:
        while parent[pmcid] != pmcid:
            parent[pmcid] = parent[parent[pmcid]]
            pmcid = parent[pmcid]
        return pmcid

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    resource_owner: dict[str, str] = {}
    for row in df.itertuples(index=False):
        pmcid = str(row.PMCID)
        for resource_id in get_resource_ids(row.NER_Tags):
            owner = resource_owner.setdefault(resource_id, pmcid)
            union(pmcid, owner)

    components: dict[str, set[str]] = {}
    for pmcid in pmcids:
        components.setdefault(find(pmcid), set()).add(pmcid)
    return sorted(components.values(), key=lambda component: (-len(component), min(component)))


def get_dataframe_resource_ids(df: pd.DataFrame) -> set[str]:
    resource_ids: set[str] = set()
    for tags in df["NER_Tags"]:
        resource_ids.update(get_resource_ids(tags))
    return resource_ids


def split_by_pmcid_and_resource(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    keep_pmcid: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError(f"Split ratios must be positive: {ratios}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")

    components = get_document_tool_components(df)
    sentence_counts = {
        str(pmcid): int(count)
        for pmcid, count in df.groupby("PMCID").size().items()
    }
    rng = random.Random(random_seed)
    rng.shuffle(components)
    components.sort(
        key=lambda component: sum(sentence_counts[pmcid] for pmcid in component),
        reverse=True,
    )

    total_sentences = len(df)
    targets = {
        name: total_sentences * ratio for name, ratio in ratios.items()
    }
    split_components: dict[str, list[set[str]]] = {
        name: [] for name in ratios
    }
    split_sizes = {name: 0 for name in ratios}

    for component in components:
        component_size = sum(sentence_counts[pmcid] for pmcid in component)
        eligible = [
            name
            for name in ratios
            if split_sizes[name] + component_size <= targets[name]
        ]
        candidates = eligible if eligible else list(ratios)
        selected = min(
            candidates,
            key=lambda name: (
                split_sizes[name] / targets[name],
                split_sizes[name],
                name,
            ),
        )
        split_components[selected].append(component)
        split_sizes[selected] += component_size

    split_pmcids = {
        name: set().union(*assigned) if assigned else set()
        for name, assigned in split_components.items()
    }
    if split_pmcids["train"] & split_pmcids["validation"]:
        raise ValueError("PMCID overlap between training and validation splits")
    if split_pmcids["train"] & split_pmcids["test"]:
        raise ValueError("PMCID overlap between training and test splits")
    if split_pmcids["validation"] & split_pmcids["test"]:
        raise ValueError("PMCID overlap between validation and test splits")

    def select(name: str) -> pd.DataFrame:
        return df[df["PMCID"].astype(str).isin(split_pmcids[name])].copy()

    raw_splits = {name: select(name) for name in ratios}
    resource_ids = {
        name: get_dataframe_resource_ids(split) for name, split in raw_splits.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = resource_ids[left] & resource_ids[right]
        if overlap:
            raise ValueError(
                f"Resource-ID overlap between {left} and {right}: "
                f"{sorted(overlap)[:10]}"
            )
    if sum(len(split) for split in raw_splits.values()) != len(df):
        raise ValueError("Rows were lost during dataset splitting")

    def finalise(split: pd.DataFrame) -> pd.DataFrame:
        if not keep_pmcid:
            split = split.drop(columns=["PMCID"])
        return split.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return (
        finalise(raw_splits["train"]),
        finalise(raw_splits["validation"]),
        finalise(raw_splits["test"]),
    )


def split_by_pmcid(
    df: pd.DataFrame,
    val_test_size: float = 0.4,
    test_ratio_of_remainder: float = 0.4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test with PMCID-level leakage prevention.

    Splits unique PMCIDs first, then assigns all rows per PMCID to the same split.
    Returns (train_df, val_df, test_df) without PMCID column, shuffled.
    """
    pmcids = sorted(df["PMCID"].unique())
    train_ids, rem_ids = train_test_split(
        pmcids, test_size=val_test_size, random_state=random_seed, shuffle=False
    )
    val_ids, test_ids = train_test_split(
        rem_ids, test_size=test_ratio_of_remainder, random_state=random_seed, shuffle=False
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    def _finalise(split: pd.DataFrame) -> pd.DataFrame:
        return (
            split.drop(columns=["PMCID"])
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )

    return (
        _finalise(df[df["PMCID"].isin(train_set)]),
        _finalise(df[df["PMCID"].isin(val_set)]),
        _finalise(df[df["PMCID"].isin(test_set)]),
    )
