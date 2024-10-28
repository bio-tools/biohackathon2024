import pandas as pd

def load_biotools_pub(path: str) -> pd.DataFrame:
    """
    Loads the BioTools publication data.

    Returns:
    pd.DataFrame: The BioTools publication data.
    """
    return pd.read_csv(path, sep="\t")
