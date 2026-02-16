import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_pmcid(
    df: pd.DataFrame,
    val_test_size: float = 0.4,
    test_ratio_of_remainder: float = 0.4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test with PMCID-level leakage prevention.

    Sorts by PMCID before splitting to ensure reproducibility.
    Returns (train_df, val_df, test_df) without PMCID column, shuffled.
    """
    df = df.sort_values(by="PMCID").reset_index(drop=True)
    train_df, remainder = train_test_split(
        df, test_size=val_test_size, random_state=random_seed, shuffle=False
    )
    remainder = remainder[~remainder["PMCID"].isin(train_df["PMCID"])].reset_index(drop=True)
    val_df, test_df = train_test_split(
        remainder, test_size=test_ratio_of_remainder, random_state=random_seed, shuffle=False
    )
    test_df = test_df[~test_df["PMCID"].isin(train_df["PMCID"])].reset_index(drop=True)

    def _finalise(split: pd.DataFrame) -> pd.DataFrame:
        return (
            split.drop(columns=["PMCID"])
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )

    return _finalise(train_df), _finalise(val_df), _finalise(test_df)
