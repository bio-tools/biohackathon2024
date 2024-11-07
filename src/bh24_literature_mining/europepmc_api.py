import csv
from dataclasses import dataclass
import json
import random
import re
import pandas as pd
import requests
from typing import Dict, List, Optional, Set
import math
from sentence_splitter import SentenceSplitter
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from pathlib import Path
import pathlib



from bh24_literature_mining.biotools import Tool_entry
from bh24_literature_mining.utils import parse_to_bool
import nltk
from nltk.tokenize import wordpunct_tokenize

# Ensure NLTK is installed and the tokenizer is available
# nltk.download('punkt')

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
                dat[i]["articles"] = [Article.dict_to_article(x) for x in tool["articles"]]
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
            Base URL for the Europe PMC API, by default "https://www.ebi.ac.uk/europepmc/webservices/rest/search".
        """
        self.base_url = base_url

    def get_data(
        self,
        query: str,
        result_type: str = "lite",
        page_size: int = 1000,
        format: str = "json",
        page_limit: int = 9,
    ) -> List[Article]:
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
            if (
                not next_cursor_mark
                or cursor_mark == next_cursor_mark
                or counter >= page_limit
            ):
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
                isOpenAccess=parse_to_bool(item.get("isOpenAccess")),
                inEPMC=parse_to_bool(item.get("inEPMC")),
                citedByCount=item.get("citedByCount"),
                pubType=item.get("pubType"),
            )
            articles.append(article)
        return articles

    def search_mentions(self, tool_name: str, article_limit = None, topics: str = None) -> List[Article]:
        """Searches for mentions of a specific tool using the Europe PMC API.

        Parameters
        ----------
        tool_name : str
            The name of the tool to search for.
        topics : bool
            Whether to use the tool EDAM topics as additional keywords.

        Returns
        -------
        List[Article]
            List of Article objects for the specified tool query.
        """
        if topics:
            query = f'"{tool_name}" AND {topics}'
        else:
            query = f'"{tool_name}"'
    
        if article_limit:
            page_limit = min(article_limit, 100)
            page_size = -(-article_limit // page_limit)  # This rounds up the division
            return self.get_data(query=query + " OPEN_ACCESS:y IN_EPMC:y", page_size=page_size, page_limit=page_limit)[:article_limit]
        else:
            return self.get_data(query=query + " OPEN_ACCESS:y IN_EPMC:y")

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
        return self.get_data(
            query=query + " OPEN_ACCESS:y IN_EPMC:y", result_type="core"
        )

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

    def get_mentions_for_tools(
        self, tools=pd.DataFrame, use_topics=False
    ) -> List[dict]:
        """Searches for articles that mention a list of tools using the Europe PMC API keyword search.
        Provides a list of dictionaries with the tool name and the list of mentioning articles  as
        article objects.

        Parameters
        ----------
        tools : DataFrame
            DataFrame with name: tool name, biotoolsID: bio.tools ID, pubmedid: PubMedID, pubmedcid: PubMedCentralID, link: link to fulltext xml
        use_topics : bool
            Whether to use the tool EDAM topics as additional keywords.

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
                tool_cites = []
                topics = row["EDAM_topics"]
                if use_topics and not str(topics) == "nan" and not str(topics) == "":
                    # Separate comma-separated EDAM_topics string into list
                    topics = row["EDAM_topics"].split(", ")
                    # embed each topics into quotes
                    topics = [f'"{topic}"' for topic in topics]
                    topics = "(" + " OR ".join(topics) + ")"
                    tool_cites = self.search_mentions(name, topics)
                else:
                    tool_cites = self.search_mentions(name, "")
                if len(tool_cites) > 0:
                    biotools_cites.append(
                        {"name": name, "pubmedid": pubmedid, "articles": tool_cites}
                    )

        return biotools_cites

    def get_relevant_paragraphs(self, pmcid: str, tool_name: str):
        """
        Retrieves paragraphs from the full text of an article that contain specific sentences.
        """
        relevant_paragraphs = []
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "lxml-xml")
            p_tags = soup.find_all("p")

            for tag in p_tags:
                paragraph_text = tag.get_text()
                if tool_name.lower() in paragraph_text.lower():
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

def find_sentences_with_substring(string_list: List[str], substring: str, limit: int = 3) -> List[str]:
    """
    Finds random sentences containing a specific substring in a list of strings.

    Parameters
    ----------
    string_list : List[str]
        List of sentences (strings) to search.
    substring : str
        Substring to search for.
    limit : int, optional
        The maximum number of sentences to retrieve, by default 3.
    
    Returns
    -------
    List[str]
        List of randomly selected sentences containing the substring, up to the specified limit.
    """
    all_sentences = []

    # Split each text into sentences and collect them in a single list
    for text in string_list:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    
    # Shuffle sentences before searching
    random.shuffle(all_sentences)

    # Search for sentences containing the substring and return as soon as the limit is reached
    matching_sentences = []
    for sentence in all_sentences:
        if substring.lower() in sentence.lower():
            matching_sentences.append(sentence.replace('\n', ' '))
            if len(matching_sentences) == limit:
                break
    
    return matching_sentences

def identify_tool_mentions_in_sentences(pmcid: str, tool: Tool_entry, paragraphs: List[str], limit: int = 3) -> List[List[str]]:
    """
    Identifies tool mentions in sentences.

    Parameters
    ----------
    pmcid : str
        The PMC ID of the article.
    tool : Tool_entry
        The Tool_entry object.
    paragraphs : List[str]
        List of paragraphs from the article.
    limit : int, optional
        The maximum number of sentences to retrieve, by default 3.

    Returns
    -------
    List[List[str]]
        List of lists containing the PMCID, sentence, NER tags, and topics.
    """
    sentences_data: Dict[str, Set] = {}
    sentences = find_sentences_with_substring(paragraphs, tool.name, limit)

    for sentence in sentences:
        if sentence:
            token = tool.name
            # Escape token to handle special regex characters
            pattern = re.escape(token)
            # Find all occurrences of the token in the sentence
            matches = re.finditer(pattern, sentence, flags=re.IGNORECASE)

            for match in matches:
                start_span = match.start()
                end_span = match.end()
                if sentence not in sentences_data:
                    sentences_data[sentence] = set()
                sentences_data[sentence].add((start_span, end_span, token, tool.biotools_id))

     # Sort sentences by the position of the first mention of the tool name
    sorted_sentences = sorted(
        sentences_data.items(),
        key=lambda item: min(tag[0] for tag in item[1])
    )
    
    # Prepare the final result
    result = [
        [pmcid, sentence, list(ner_tags), tool.topics_str]
        for sentence, ner_tags in sentences_data.items()
    ]

    return result

def identify_tool_mentions_using_europepmc(biotools: List[Tool_entry], article_limit: int=1, sentences_per_article: int = 3) -> pd.DataFrame:
    """
    Identifies tool mentions in sentences using the Europe PMC API.

    Parameters
    ----------
    biotools : List[Tool_entry]
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
    for tool in biotools:
        # Call bio.tools query and get a list of Article objects
        
        biotools_articles: List[Article] = client.search_mentions(tool.name, article_limit = article_limit, topics=tool.disjoint_topics())

        if len(biotools_articles) == 0:
            print("No articles found", tool.name)
            continue
        for article in biotools_articles:
            # per each article, get relevant paragraphs
            relevant_paragraphs = client.get_relevant_paragraphs(article.pmcid, tool.name)
            if len(relevant_paragraphs) == 0:
                print("No relevant paragraphs found", tool.name)
                continue
            
            result = identify_tool_mentions_in_sentences(article.pmcid, tool, relevant_paragraphs, sentences_per_article)
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

def filter_trainning_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the input DataFrame to only include rows that are ground truth.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    p = Path().cwd().parents[0]
    if not p/"generated_data":
        p.mkdir(parents=True, exist_ok=True)

    filtered_df = dataframe[dataframe["True?"]].drop(columns=["True?", "False?"])
    filtered_df.to_csv(p / "generated_data" / "filtered_data.csv", index=False)

    return filtered_df

def find_sub_span(token_span, entity_span):
    if token_span[0] < entity_span[1] and token_span[1] > entity_span[0]:
        return max(token_span[0], entity_span[0]), min(token_span[1], entity_span[1])
    return None

def convert_to_iob(texts, ner_tags_list):
    results = []

    for text, ner_tags in zip(texts, ner_tags_list):
        # Tokenize using NLTK's wordpunct_tokenizer
        tokens = wordpunct_tokenize(text)
        token_spans = []
        current_idx = 0

        # Calculate token spans based on the original text
        for token in tokens:
            start_idx = text.find(token, current_idx)
            end_idx = start_idx + len(token)
            token_spans.append((start_idx, end_idx))
            current_idx = end_idx

        iob_tags = ['O'] * len(tokens)

        for start, end, entity, entity_type in sorted(ner_tags, key=lambda x: x[0]):
            entity_flag = False  # Flag to indicate if we are inside an entity
            for i, token_span in enumerate(token_spans):
                if find_sub_span(token_span, (start, end)):
                    if not entity_flag:  # If it's the start of an entity
                        iob_tags[i] = 'B-' + entity_type
                        entity_flag = True
                    elif iob_tags[i] == 'O':  # Continue tagging inside of the entity
                        iob_tags[i] = 'I-' + entity_type
                else:
                    entity_flag = False  # Reset flag when we're no longer in an entity

        results.append(list(zip(tokens, iob_tags)))
    return results

def convert_to_IOB_format_from_df(dataframe, output_folder, filename, batch_size=500):
    # Prepare data for batch processing
    data = [(row['Sentence'], row['NER_Tags']) for index, row in dataframe.iterrows()]

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    result_path = os.path.join(output_folder, filename)

    with open(result_path, 'w', newline='\n') as f1:
        train_writer = csv.writer(f1, delimiter='\t', lineterminator='\n')

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i:i+batch_size]
            sentences, ner_tags_batch = zip(*batch)

            # Convert to IOB format
            batch_results = convert_to_iob(sentences, ner_tags_batch)

            for tagged_tokens in batch_results:
                for each_token in tagged_tokens:
                    train_writer.writerow(list(each_token))
                train_writer.writerow('')