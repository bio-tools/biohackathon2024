from dataclasses import dataclass
import re
import csv
import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

from bh24_literature_mining.biotools import Tool_entry
from bh24_literature_mining.utils import parse_to_bool
from sentence_splitter import SentenceSplitter
from sklearn.model_selection import train_test_split
import pandas as pd

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

    def __init__(self, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search"):
        """Initializes the EuropePMCClient with a base URL for the API.

        Parameters
        ----------
        base_url : str, optional
            Base URL for the Europe PMC API, by default "https://www.ebi.ac.uk/europepmc/webservices/rest/search".
        """
        self.base_url = base_url

    def get_data(self, query: str, result_type: str = "lite", page_size: int = 1, format: str = "json", page_limit : int = 3) -> List[Article]:
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
                "format": format
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raises an error for bad status codes
            
            json_response = response.json()
            articles.extend(self._parse_articles(json_response))  # Add batch to main list
            
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
                pubType=item.get("pubType")
            )
            articles.append(article)
        return articles
    

    def search_mentions(self, tool_name: str, article_limit = None) -> List[Article]:
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
        if article_limit:
            page_limit = min(article_limit, 100)
            page_size = -(-article_limit // page_limit)  # This rounds up the division
            return self.get_data(query=tool_name + " OPEN_ACCESS:y IN_EPMC:y", page_size=page_size, page_limit=page_limit)
        else:
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

    def get_relevant_paragraphs(self, pmcid: str, tool_name: str):
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
                if tool_name in paragraph_text:
                    relevant_paragraphs.append(paragraph_text)

            return relevant_paragraphs
        else:
            return []
        
def segment_sentences_spacy(paragraphs:List[str], substring):
    if not paragraphs:
        return None
    all_sentences = []
    for parahgraph in paragraphs:
        splitter = SentenceSplitter(language='en')
        sentences = splitter.split(parahgraph)
        for sentence in sentences:
            if substring in sentence:
                all_sentences.append(sentence)
    return all_sentences

def find_sentence_with_substring(string_list, substring):
    all_sentences = []
    for text in string_list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if substring.lower() in sentence.lower():
                all_sentences.append(sentence.replace('\n', ' '))
    return all_sentences

def identify_tool_mentions_in_sentences(pmcid:str, tool: Tool_entry, paragraphs:List[str]):
    sentences_data = {}
    sentences = find_sentence_with_substring(paragraphs, tool.name)
    for sentence in sentences:
        if sentence:
            token = tool.name
            start_span = sentence.find(token)
            end_span = start_span + len(token)

            if start_span != -1:  # Ensure the token is found in the sentence
                if sentence not in sentences_data:
                    sentences_data[sentence] = set()

                sentences_data[sentence].add((start_span, end_span, token, tool.biotool_id))

    return [[pmcid, sentence, list(ner_tags), tool.topics] for sentence, ner_tags in sentences_data.items()]

def identify_tool_mentions_using_europepmc(biotools: List[Tool_entry], article_limit: int=1) -> pd.DataFrame:
    """
    Identifies tool mentions in sentences using the Europe PMC API.

    Parameters
    ----------
    biotools : List[Tool_entry]
        List of Tool_entry objects.
    
    article_limit : int, optional
        The maximum number of articles to retrieve for each tool, by default 1.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the tool mentions.
    """
    results_list = []
    client = EuropePMCClient()
    for tool in biotools:
        # Call bio.tools query and get a list of Article objects
        
        biotools_articles = client.search_mentions(tool.name, article_limit = article_limit)

        if len(biotools_articles) == 0:
            continue
        first_article = biotools_articles[0]

        relevant_parahraphs = client.get_relevant_paragraphs(first_article.pmcid, tool.name)
        if len(relevant_parahraphs) == 0:
            print("No relevant paragraphs found", tool.name)
            continue
        
        tool.get_topics()

        result = identify_tool_mentions_in_sentences(first_article.pmcid, tool, relevant_parahraphs)
        results_list.extend(result)

    result_df = pd.DataFrame(results_list, columns=["PMCID", "Sentence", "NER_Tags", "Topics"])
    result_df = result_df.explode("NER_Tags").drop_duplicates()

    return result_df

def split_train_test_dev(dataframe: pd.DataFrame, train_size=0.7, dev_size=0.5, random_state=42):
    """
    Splits the input dataframe into train, test, and dev sets.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to split.
        train_size (float): Proportion of the data to use for training. Default is 0.7.
        dev_size (float): Proportion of the test_dev split to use for dev (e.g., 0.5 for 50%). Default is 0.5.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Test set.
        dev_df (pd.DataFrame): Development/validation set.
    """
    # First split: train and test_dev (test + dev combined)
    train_df, test_dev_df = train_test_split(dataframe, train_size=train_size, random_state=random_state)
    
    # Second split: split test_dev into test and dev
    test_df, dev_df = train_test_split(test_dev_df, test_size=dev_size, random_state=random_state)
    
    return train_df, test_df, dev_df
