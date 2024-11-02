import pandas as pd
import json, zipfile

def load_biotools_pub(path: str) -> pd.DataFrame:
    """
    Loads the BioTools publication data.

    Returns:
    pd.DataFrame: The BioTools publication data.
    """
    return pd.read_csv(path, sep="\t")


def load_biotools_from_json(path: str) -> list:
    """Load tools from a JSON file.
    
    Returns:
    list: A list of tools.
    """
    return json.load(path)


def load_biotools_from_zip(path: str, file_name: str) -> list:
    """Load tools from a ZIP file.
    
    Returns:
    list: A list of tools.
    """
    with zipfile.ZipFile(path) as z:
        with z.open(file_name) as f:
            return json.load(f)
        

def save_to_json(tools, json_path):
    """Save tools to a JSON file."""
    with open(json_path, 'w') as file:
        json.dump(tools, file, indent=4)
