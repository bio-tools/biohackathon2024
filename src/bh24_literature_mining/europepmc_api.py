from dataclasses import dataclass
import requests
from typing import List, Optional
import pandas as pd
import math

from bs4 import BeautifulSoup

from bh24_literature_mining.utils import parse_to_bool

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

    inEPMC: Optional[bool] = None
    

class EuropePMCClient:
    """Client for interacting with the Europe PMC API."""

    def __init__(
        self, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ):
        """Initializes the EuropePMCClient with a base URL for the API.

        Parameters
        ----------
        base_url : str, optional
            Base URL for the Europe PMC API, by default "https://www.ebi.ac.uk/europepmc/webservices/rest/search".
        """
        self.base_url = base_url

    def get_data(self, query: str, result_type: str = "lite", page_size: int = 1000, format: str = "json", page_limit : int = 9) -> List[Article]:
        """
        Makes an API request and retrieves all pages by looping until all results are fetched.
        
        Parameters
        ----------
        query : str
            The query string for the search.
        result_type : str, optional
            The type of result to retrieve, by default "lite".
        page_size : int, optional
            The number of results to retrieve per page, by default 25.
        format : str, optional
            The format of the response, by default "json".
        page_limit : int, optional
            The maximum number of pages to retrieve, by default 3.
        
        Returns
        -------
        List[Article]
            List of all Article objects from the API response.
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
                "format": format,
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raises an error for bad status codes

            json_response = response.json()
            articles.extend(
                self._parse_articles(json_response)
            )  # Add batch to main list

            # Update cursor_mark to the nextCursorMark from the response
            next_cursor_mark = json_response.get("nextCursorMark")
            if not next_cursor_mark or cursor_mark == next_cursor_mark or counter >= page_limit:
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
                id=item.get("id"),
                title=item.get("title"),
                authorString=item.get("authorString"),
                pubYear=item.get("pubYear"),
                journalTitle=item.get("journalTitle"),
                pubDate=item.get("pubDate"),
                doi=item.get("doi"),
                pmcid=item.get("pmcid"),
                pmid=item.get("pmid"),
                isOpenAccess= parse_to_bool(item.get("isOpenAccess")),
                inEPMC=parse_to_bool(item.get("inEPMC")),
                citedByCount=item.get("citedByCount"),
                pubType=item.get("pubType"),
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

        return self.get_data(query=tool_name + " OPEN_ACCESS:y IN_EPMC:y")

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

    def get_cites_for_tools(self, tools=pd.DataFrame) -> List[dict]:
        """Searches for articles that cite a list of tools using the Europe PMC API.
        Provides a list of dictionaries with the tool name and the list of citing articles as
        article objects.

        Parameters
        ----------
        tools : DataFrame
            DataFrame with name: tool name, biotoolsID: bio.tools ID, pubmedid: PubMedID, pubmedcid: PubMedCentralID, link: link to fulltext xml

        Returns
        -------
        List[Article]
            List of dictionaries with name of tools, pubmediid and list of Article objects for the citations query.
        """
        biotools_cites = []
        print("Total number of tools: ", len(tools.index))

        for index, row in tools.iterrows():
            if index > -1:
                name = row["name"]
                pubmedid = row["pubmedid"]
                if not math.isnan(pubmedid):
                    pubmedid = round(pubmedid)
                link = row["link"]
                print(
                    f"Iter: {index}, Name: {name}, PubMed ID: {pubmedid}, Link: {link}"
                )
                # Call bio.tools query and get a list of Article objects
                tool_cites = self.search_cites(pubmedid)
                if len(tool_cites) > 0:
                    biotools_cites.append(
                        {"name": name, "pubmedid": pubmedid, "articles": tool_cites}
                    )

        return biotools_cites

    def get_mentions_for_tools(self, tools=pd.DataFrame) -> List[dict]:
        """Searches for articles that mention a list of tools using the Europe PMC API keyword search.
        Provides a list of dictionaries with the tool name and the list of mentioning articles  as
        article objects.

        Parameters
        ----------
        tools : DataFrame
            DataFrame with name: tool name, biotoolsID: bio.tools ID, pubmedid: PubMedID, pubmedcid: PubMedCentralID, link: link to fulltext xml

        Returns
        -------
        List[Article]
            List of dictionaries with name of tools, pubmediid and list of Article objects for the citations query.
        """

        biotools_cites = []
        print("Total number of tools: ", len(tools.index))

        for index, row in tools.iterrows():
            if index > -1:
                name = row["name"]
                pubmedid = row["pubmedid"]
                if not math.isnan(pubmedid):
                    pubmedid = round(pubmedid)
                link = row["link"]
                print(
                    f"Iter: {index}, Name: {name}, PubMed ID: {pubmedid}, Link: {link}"
                )
                # Call bio.tools query and get a list of Article objects
                tool_cites = self.search_mentions(name)
                if len(tool_cites) > 0:
                    biotools_cites.append(
                        {"name": name, "pubmedid": pubmedid, "articles": tool_cites}
                    )

        return biotools_cites
    def get_relevant_paragraphs(self, pmcid):
        """
        Retrieves paragraphs from the full text of an article that contain specific sentences.
        """
        relevant_paragraphs = []
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml-xml')
            p_tags = soup.find_all('p')
            
            for tag in p_tags:
                paragraph_text = tag.get_text()
                # if any(partial_sentence in paragraph_text for partial_sentence in partial_sentences):
                relevant_paragraphs.append(paragraph_text)

            return relevant_paragraphs
        else:
            return None
