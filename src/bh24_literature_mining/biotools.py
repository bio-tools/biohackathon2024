import requests
from typing import List
from bh24_literature_mining.utils import load_biotools_pub


class Tool_entry:
    """
    Data structure for storing tool information.
    """

    def __init__(
        self,
        biotools_id: str,
        name: str,
        topics: str,
        pubmedid: str,
        pubmedcid: str,
        link: str,
    ):
        self.biotools_id = biotools_id
        self.name = name
        self.topics_str = topics
        if topics is None or isinstance(topics, float) or topics == "":
            self.topics_list = []
        else:
            self.topics_list = topics.split(", ")
        self.pubmedid = pubmedid
        self.pubmedcid = pubmedcid
        self.link = link

    def disjoint_topics(self) -> str:
        return "(" + " OR ".join(self.topics_list) + ")"


def get_tool_by_id(biotoolsid: str) -> dict:
    """
    Get the tool information from BioTools API by ID.

    Parameters
    ----------
    biotoolsid : str
        The BioTools ID of the tool.

    Returns
    -------
    dict
        The JSON response from the BioTools API.
    """
    base_url = f"https://bio.tools/api/tool/{biotoolsid}"
    params = {"format": "json"}

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Check for request errors
    return response.json()


def get_biotools(path_to_file: str, limit: int = 400) -> List[Tool_entry]:
    """
    Load the BioTools data and create a list of Tool_entry objects.

    Parameters
    ----------
    pat_to_file : str
        The path to the tsv file containing the BioTools data.

    limit : int, optional
        The maximum number of tools to load. Default is 400.

    Returns
    -------
    List[Tool_entry]
        A list of Tool_entry objects.
    """

    tools = load_biotools_pub(path_to_file)

    # Initialize set to track unique names and a list to store Biotool objects
    tools_lower = set()
    unique_biotools = []

    # Iterate over the rows in the DataFrame
    for _, row in tools.iterrows():
        name = row["name"]
        biotools_id = row["biotoolsID"]
        pubmedid = row["pubmedid"]
        pubmedcid = row["pubmedcid"]
        link = row["link"]
        topics = row["EDAM_topics"]
        name_lower = name.lower()

        # Only add unique tools based on name
        if name_lower not in tools_lower:
            tools_lower.add(name_lower)  # Add to the set to track uniqueness
            # Create a Biotool object and add it to the list
            biotool = Tool_entry(biotools_id, name, topics, pubmedid, pubmedcid, link)
            unique_biotools.append(biotool)

    return unique_biotools
