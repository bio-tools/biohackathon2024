import pandas as pd
from sklearn.model_selection import train_test_split


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
