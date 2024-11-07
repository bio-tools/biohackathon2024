import pandas as pd
import json, zipfile
from bh24_literature_mining.europepmc_api import Article


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

    Parameters:
    path (str): The path to the ZIP file.
    file_name (str): The name of the file to load from the zip file. (e.g., "biotools.json", "proteomics_tools.json")    

    Returns:
    list: A list of tools.
    """
    with zipfile.ZipFile(path) as z:
        with z.open(file_name) as f:
            return json.load(f)


def save_to_json(tools, json_path: str):
    """Save tools to a JSON file."""
    with open(json_path, "w") as file:
        json.dump(tools, file, indent=4, default=lambda x: x.__dict__)


def read_cites_from_json(path: str) -> list[dict]:
    """Reads the list of tools and their citing articles from a JSON file.

    Parameters
    ----------
    path : str
            Path to the JSON file.

    Returns
    -------
    list[dict]
        List of dictionaries with the tool name and the list of citing articles.
    """
    with open(path, "r") as file:
        dat = json.load(file)
        for i, tool in enumerate(dat):
            dat[i]["articles"] = [Article.dict_to_article(x) for x in tool["articles"]]
        return dat


def parse_to_bool(value: str) -> bool:
    """Parses a string value to a boolean.

    Parameters
    ----------
    value : str
        The value to parse.

    Returns
    -------
    bool
        The boolean value.
    """
    return value.lower() == "true" or value == "1" or value == "t"
