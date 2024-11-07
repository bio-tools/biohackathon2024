
from dataclasses import dataclass
from typing import List
import requests
from bh24_literature_mining.utils import load_biotools_pub

@dataclass
class Tool_entry:
    """
    Data structure for storing tool information.
    """

    biotool_id: str
    name: str
    topics: List[str] = None


    def get_topics(self):
        self.topics = "N/A"
        json = get_tool_by_id(self.biotool_id)
        json_topics = json.get("topic")

        terms = []
        # Loop through each dictionary in the topics list
        for topic in json_topics:
            # Access the 'term' value from the current dictionary and append it to the terms list
            terms.append(topic['term'])
        self.topics = ", ".join(terms)

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



def get_biotools(path_to_file: str) -> List[Tool_entry]:
    """
    Load the BioTools data and create a list of Tool_entry objects.
    
    Parameters
    ----------
    pat_to_file : str
        The path to the tsv file containing the BioTools data.
        
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
        name_lower = name.lower()

        # Only add unique tools based on name
        if name_lower not in tools_lower:
            tools_lower.add(name_lower)  # Add to the set to track uniqueness
            # Create a Biotool object and add it to the list
            biotool = Tool_entry(biotool_id=biotools_id, name=name)
            unique_biotools.append(biotool)

    return unique_biotools