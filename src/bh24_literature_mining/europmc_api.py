from dataclasses import dataclass
import requests
from typing import List, Optional

@dataclass
class Article:
    id: str
    title: str
    authorString: str
    pubYear: str
    journalTitle: str
    pubDate: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    pmid: Optional[str] = None

class EuropePMCClient:


    def __init__(self, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search"):
        self.base_url = base_url


    def get_data(self, query: str, result_type: str = "lite", cursor_mark: str = "*", page_size: int = 25, format: str = "json") -> List[Article]:
        """
        Makes an API request and returns a list of Article objects.
        :param query: The query string for the search.
        :param result_type: The type of result to retrieve (default is "lite").
        :param cursor_mark: The cursor mark for pagination (default is "*").
        :param page_size: The number of results to retrieve per page (default is 25).
        :param format: The format of the response (default is "json").
        :return: List of Article objects.
        """
        params = {
            "query": query,
            "resultType": result_type,
            "cursorMark": cursor_mark,
            "pageSize": page_size,
            "format": format
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()  # Raises an error for bad status codes
        json_response = response.json()
        articles = self._parse_articles(json_response)
        return articles
    

    def _parse_articles(self, json_response) -> List[Article]:
        """
        Parses the JSON response into a list of Article objects.
        :param json_response: JSON response from the API.
        :return: List of Article objects.
        """
        articles = []
        for item in json_response.get("resultList", {}).get("result", []):
            article = Article(
                id=item.get("id", ""),
                title=item.get("title", ""),
                authorString=item.get("authorString", ""),
                pubYear=item.get("pubYear", ""),
                journalTitle=item.get("journalTitle", ""),
                pubDate=item.get("pubDate"),
                doi=item.get("doi"),
                pmcid=item.get("pmcid"),
                pmid=item.get("pmid")
            )
            articles.append(article)
        return articles
    

    def search_mentions(self, tool_name: str) -> List[Article]:
        """
        Calls the API with a specific query for 'bio.tools' and returns a list of Article objects.

        Parameters:

        tool_name (str): The name of the tool to search for.

        Returns:
            List of Article objects for the 'bio.tools' query.
        """
        return self.get_data(query=tool_name)
    

    def search_cites(self, pmid: str) -> List[Article]:
        """
        Calls the API with a query to search for articles citing a specific PubMed ID and returns a list of Article objects.
        :param pmid: PubMed ID to search citations for.
        :return: List of Article objects for the 'cites' query.
        """
        query = f"cites:{pmid}_MED"
        return self.get_data(query=query, result_type="core")
    


# Usage example
if __name__ == "__main__":
    client = EuropePMCClient()
    # Call bio.tools query and get a list of Article objects
    biotools_articles = client.search_mentions("PeptideProphet")
    print("Bio.tools articles:", biotools_articles)
    # Call cites query with a specific PubMed ID and get a list of Article objects
    cites_articles = client.search_cites("32109013")
    print("Cites articles:", cites_articles)








