
from dataclasses import dataclass
from typing import List
import requests

@dataclass
class Tool_entry:

    biotool_id: str
    name: str
    topics: List[str] = None


    def get_topics(self):
        json = get_tool_by_id(self.biotool_id)
        json_topics = json.get("topic")

        terms = []
        # Loop through each dictionary in the topics list
        for topic in json_topics:
            # Access the 'term' value from the current dictionary and append it to the terms list
            terms.append(topic['term'])
        self.topics = terms

def get_tool_by_id(biotoolsid: str) -> dict:
    base_url = f"https://bio.tools/api/tool/{biotoolsid}"
    params = {"format": "json"}

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Check for request errors
    return response.json()