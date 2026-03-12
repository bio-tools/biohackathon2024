import json
import logging
import math
import random
import re
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_splitter import SentenceSplitter
from tqdm import tqdm

from bh24_literature_mining.biotools import Tool_entry
from bh24_literature_mining.utils import parse_to_bool

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Data structure for storing article information."""

    id: str
    title: str
    authorString: str
    pubYear: str
    journalTitle: str
    pubDate: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    pmid: str | None = None
    isOpenAccess: bool | None = None
    citedByCount: int | None = None
    pubType: str | None = None

    inEPMC: bool | None = None

    @staticmethod
    def dict_to_article(article_dict: dict):
        return Article(
            id=article_dict.get("id"),
            title=article_dict.get("title"),
            authorString=article_dict.get("authorString"),
            pubYear=article_dict.get("pubYear"),
            journalTitle=article_dict.get("journalTitle"),
            pubDate=article_dict.get("pubDate"),
            doi=article_dict.get("doi"),
            pmcid=article_dict.get("pmcid"),
            pmid=article_dict.get("pmid"),
            isOpenAccess=article_dict.get("isOpenAccess"),
            inEPMC=article_dict.get("inEPMC"),
            citedByCount=article_dict.get("citedByCount"),
            pubType=article_dict.get("pubType"),
        )

    @staticmethod
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
                dat[i]["articles"] = [
                    Article.dict_to_article(x) for x in tool["articles"]
                ]
            return dat


class EuropePMCClient:
    """Client for interacting with the Europe PMC API."""

    def __init__(
        self, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ):
        """Initializes the EuropePMCClient with a base URL for the API.

        Parameters
        ----------
        base_url : str, optional
            Base URL for the Europe PMC API, by default
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search".
        """
        self.base_url = base_url

    def get_data(
        self,
        query: str,
        result_type: str = "lite",
        page_size: int = 1000,
        response_format: str = "json",
        page_limit: int = 9,
    ) -> list[Article]:
        """
        Makes an API request and retrieves all pages by looping
        until all results are fetched.

        Parameters
        ----------
        query : str
            The query string for the search.
        result_type : str, optional
            The type of result to retrieve, by default "lite".
        page_size : int, optional
            The number of results to retrieve per page, by default 1000.
        response_format : str, optional
            The format of the response, by default "json".
        page_limit : int, optional
            The maximum number of pages to retrieve, by default 9.

        Returns
        -------
        list[Article]
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
                "format": response_format,
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            json_response = response.json()
            articles.extend(self._parse_articles(json_response))

            next_cursor_mark = json_response.get("nextCursorMark")
            if (
                not next_cursor_mark
                or cursor_mark == next_cursor_mark
                or counter >= page_limit
            ):
                break
            cursor_mark = next_cursor_mark

        return articles

    def _parse_articles(self, json_response) -> list[Article]:
        """Parses the JSON response into a list of Article objects.

        Parameters
        ----------
        json_response : dict
            JSON response from the API.

        Returns
        -------
        list[Article]
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
                isOpenAccess=parse_to_bool(item.get("isOpenAccess")),
                inEPMC=parse_to_bool(item.get("inEPMC")),
                citedByCount=item.get("citedByCount"),
                pubType=item.get("pubType"),
            )
            articles.append(article)
        return articles

    def search_mentions(
        self, tool_name: str, article_limit=None, topics: str = None
    ) -> list[Article]:
        """Searches for mentions of a specific tool using the Europe PMC API.

        Parameters
        ----------
        tool_name : str
            The name of the tool to search for.
        topics : str
            EDAM topics as additional keywords.

        Returns
        -------
        list[Article]
            List of Article objects for the specified tool query.
        """
        if topics:
            query = f'"{tool_name}" AND {topics}'
        else:
            query = f'"{tool_name}"'

        if article_limit:
            page_limit = min(article_limit, 100)
            page_size = -(-article_limit // page_limit)
            return self.get_data(
                query=query + " OPEN_ACCESS:y IN_EPMC:y",
                page_size=page_size,
                page_limit=page_limit,
            )[:article_limit]
        else:
            return self.get_data(query=query + " OPEN_ACCESS:y IN_EPMC:y")

    def search_cites(self, pmid: str) -> list[Article]:
        """Searches for articles citing a specific PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID to search citations for.

        Returns
        -------
        list[Article]
            List of Article objects for the citations query.
        """
        query = f"cites:{pmid}_MED"
        return self.get_data(
            query=query + " OPEN_ACCESS:y IN_EPMC:y", result_type="core"
        )

    def get_cites_for_tools(self, tools: pd.DataFrame) -> list[dict]:
        """Searches for articles that cite a list of tools
        using the Europe PMC API. Provides a list of dictionaries
        with the tool name and the list of citing articles as article objects.

        Parameters
        ----------
        tools : DataFrame
            DataFrame with name: tool name, biotoolsID: bio.tools ID, pubmedid:
            PubMedID, pubmedcid: PubMedCentralID, link: link to fulltext xml

        Returns
        -------
        list[dict]
            List of dictionaries with name of tools, pubmediid and list of
            Article objects for the citations query.
        """
        biotools_cites = []
        logger.info("Total number of tools: %d", len(tools.index))

        for index, row in tools.iterrows():
            name = row["name"]
            pubmedid = row["pubmedid"]
            if not math.isnan(pubmedid):
                pubmedid = round(pubmedid)
            link = row["link"]
            logger.info(
                "Iter: %s, Name: %s, PubMed ID: %s, Link: %s",
                index, name, pubmedid, link,
            )
            tool_cites = self.search_cites(pubmedid)
            if tool_cites:
                biotools_cites.append(
                    {"name": name, "pubmedid": pubmedid, "articles": tool_cites}
                )

        return biotools_cites

    def get_mentions_for_tools(
        self, tools: pd.DataFrame, use_topics: bool = False
    ) -> list[dict]:
        """Searches for articles that mention a list of tools using the Europe
        PMC API keyword search. Provides a list of dictionaries
        with the tool name and the list of mentioning articles as article objects.

        Parameters
        ----------
        tools : DataFrame
            DataFrame with name: tool name, biotoolsID: bio.tools ID, pubmedid:
            PubMedID, pubmedcid: PubMedCentralID, link: link to fulltext xml
        use_topics : bool
            Whether to use the tool EDAM topics as additional keywords.

        Returns
        -------
        list[dict]
            List of dictionaries with name of tools, pubmediid and list of
            Article objects for the citations query.
        """

        biotools_cites = []
        logger.info("Total number of tools: %d", len(tools.index))

        for index, row in tools.iterrows():
            name = row["name"]
            pubmedid = row["pubmedid"]
            if not math.isnan(pubmedid):
                pubmedid = round(pubmedid)
            link = row["link"]
            logger.info(
                "Iter: %s, Name: %s, PubMed ID: %s, Link: %s",
                index, name, pubmedid, link,
            )
            tool_cites = []
            topics = row["EDAM_topics"]
            if use_topics and str(topics) != "nan" and str(topics) != "":
                topics_list = row["EDAM_topics"].split(", ")
                topics_query = "(" + " OR ".join(f'"{t}"' for t in topics_list) + ")"
                tool_cites = self.search_mentions(name, topics=topics_query)
            else:
                tool_cites = self.search_mentions(name, topics="")
            if tool_cites:
                biotools_cites.append(
                    {"name": name, "pubmedid": pubmedid, "articles": tool_cites}
                )

        return biotools_cites

    def get_relevant_paragraphs(self, pmcid: str, tool_name: str) -> list[str]:
        """
        Retrieves paragraphs from the full text of an article that
        contain specific sentences.
        """
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        response = requests.get(url)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.content, "lxml-xml")
        relevant_paragraphs = []
        for tag in soup.find_all("p"):
            paragraph_text = tag.get_text()
            if tool_name.lower() in paragraph_text.lower():
                relevant_paragraphs.append(paragraph_text)

        return relevant_paragraphs


_SENTENCE_SPLITTER = SentenceSplitter(language="en")


def segment_sentences_spacy(paragraphs: list[str], substring: str) -> list[str] | None:
    if not paragraphs:
        return None
    all_sentences = []
    for paragraph in paragraphs:
        sentences = _SENTENCE_SPLITTER.split(paragraph)
        for sentence in sentences:
            if substring in sentence:
                all_sentences.append(sentence)
    return all_sentences


def find_sentences_with_substring(
    string_list: list[str], substring: str, limit: int = 3
) -> list[str]:
    """
    Finds random sentences containing a specific substring in a list of strings.

    Parameters
    ----------
    string_list : list[str]
        List of sentences (strings) to search.
    substring : str
        Substring to search for.
    limit : int, optional
        The maximum number of sentences to retrieve, by default 3.

    Returns
    -------
    list[str]
        List of randomly selected sentences containing the substring, up to the specified limit.
    """
    all_sentences = []

    for text in string_list:
        all_sentences.extend(re.split(r"(?<=[.!?])\s+", text))

    rng = random.Random(42)
    rng.shuffle(all_sentences)

    matching_sentences = []
    for sentence in all_sentences:
        if substring.lower() in sentence.lower():
            matching_sentences.append(sentence.replace("\n", " "))
            if len(matching_sentences) == limit:
                break

    return matching_sentences


def identify_tool_mentions_in_sentences(
    pmcid: str, tool: Tool_entry, paragraphs: list[str], limit: int = 3
) -> list[list]:
    """
    Identifies tool mentions in sentences.

    Parameters
    ----------
    pmcid : str
        The PMC ID of the article.
    tool : Tool_entry
        The Tool_entry object.
    paragraphs : list[str]
        List of paragraphs from the article.
    limit : int, optional
        The maximum number of sentences to retrieve, by default 3.

    Returns
    -------
    list[list]
        List of lists containing the PMCID, sentence, NER tags, and topics.
    """
    sentences_data: dict[str, set] = {}
    sentences = find_sentences_with_substring(paragraphs, tool.name, limit)

    for sentence in sentences:
        if sentence:
            token = tool.name
            pattern = re.escape(token)
            matches = re.finditer(pattern, sentence, flags=re.IGNORECASE)

            for match in matches:
                start_span = match.start()
                end_span = match.end()
                if sentence not in sentences_data:
                    sentences_data[sentence] = set()
                sentences_data[sentence].add(
                    (start_span, end_span, token, tool.biotools_id)
                )

    sorted_sentences = sorted(
        sentences_data.items(), key=lambda item: min(tag[0] for tag in item[1])
    )

    result = [
        [pmcid, sentence, list(ner_tags), tool.topics_str]
        for sentence, ner_tags in sorted_sentences
    ]

    return result


def identify_tool_mentions_using_europepmc(
    biotools: list[Tool_entry], article_limit: int = 1, sentences_per_article: int = 3
) -> pd.DataFrame:
    """
    Identifies tool mentions in sentences using the Europe PMC API.

    Parameters
    ----------
    biotools : list[Tool_entry]
        List of Tool_entry objects.

    article_limit : int, optional
        The maximum number of articles to retrieve for each tool, by default 1.

    sentences_per_article : int, optional
        The maximum number of sentences to retrieve for each article, by default 3.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the tool mentions.
    """
    results_list = []
    client = EuropePMCClient()
    for tool in tqdm(biotools, desc="Fetching tool mentions", unit="tool"):
        biotools_articles: list[Article] = client.search_mentions(
            tool.name, article_limit=article_limit, topics=tool.disjoint_topics()
        )

        if not biotools_articles:
            logger.info("No articles found for %s", tool.name)
            continue
        for article in biotools_articles:
            relevant_paragraphs = client.get_relevant_paragraphs(
                article.pmcid, tool.name
            )
            if not relevant_paragraphs:
                logger.info("No relevant paragraphs found for %s", tool.name)
                continue

            result = identify_tool_mentions_in_sentences(
                article.pmcid, tool, relevant_paragraphs, sentences_per_article
            )
            results_list.extend(result)

    result_df = pd.DataFrame(
        results_list, columns=["PMCID", "Sentence", "NER_Tags", "Topics"]
    )
    result_df = result_df.explode("NER_Tags").drop_duplicates()

    return result_df
