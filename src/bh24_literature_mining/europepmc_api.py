from dataclasses import dataclass
import requests
from typing import List, Optional

@dataclass
class Article:
    """Data structure for storing article information."""

    id: str
    title: str
    authorString: str
    pubYear: str
    journalTitle: str
    pubDate: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    isOpenAccess: Optional[bool] = None
    citedByCount: Optional[int] = None
    pubType: Optional[str] = None
    

class EuropePMCClient:
    """Client for interacting with the Europe PMC API."""

    def __init__(self, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search"):
        """Initializes the EuropePMCClient with a base URL for the API.

        Parameters
        ----------
        base_url : str, optional
            Base URL for the Europe PMC API, by default "https://www.ebi.ac.uk/europepmc/webservices/rest/search".
        """
        self.base_url = base_url

    def get_data(self, query: str, result_type: str = "lite", page_size: int = 25, format: str = "json", page_limit : int = 3) -> List[Article]:
        """
        Makes an API request and retrieves all pages by looping until all results are fetched.
        
        :param query: The query string for the search.
        :param result_type: The type of result to retrieve (default is "lite").
        :param page_size: The number of results to retrieve per page (default is 25).
        :param format: The format of the response (default is "json").
        :return: List of all Article objects from the API response.
        """
        articles = []
        cursor_mark = "*"
        counter = 0
        while True:
            counter += 1
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
            articles.extend(self._parse_articles(json_response))  # Add batch to main list
            
            # Update cursor_mark to the nextCursorMark from the response
            next_cursor_mark = json_response.get("nextCursorMark")
            if not next_cursor_mark or cursor_mark == next_cursor_mark or counter > page_limit:
                break  # Exit loop when we've retrieved all pages
            cursor_mark = next_cursor_mark  # Move to next page

        return articles

    def _parse_articles(self, json_response) -> List[Article]:
        """Parses the JSON response into a list of Article objects.

        Parameters
        ----------
        json_response : dict
            JSON response from the API.

        Returns
        -------
        List[Article]
            List of Article objects.
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
                pmid=item.get("pmid"),
                isOpenAccess=item.get("isOpenAccess"),
                citedByCount=item.get("citedByCount"),
                pubType=item.get("pubType")
            )
            articles.append(article)
        return articles

    def search_mentions(self, tool_name: str) -> List[Article]:
        """Searches for mentions of a specific tool using the Europe PMC API.

        Parameters
        ----------
        tool_name : str
            The name of the tool to search for.

        Returns
        -------
        List[Article]
            List of Article objects for the specified tool query.
        """

        return self.get_data(query=tool_name)

    def search_cites(self, pmid: str) -> List[Article]:
        """Searches for articles citing a specific PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID to search citations for.

        Returns
        -------
        List[Article]
            List of Article objects for the citations query.
        """
        query = f"cites:{pmid}_MED"
        return self.get_data(query=query, result_type="core")

