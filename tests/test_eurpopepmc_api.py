from bh24_literature_mining.biotools import Tool_entry, get_biotools
from bh24_literature_mining.europepmc_api import (
    EuropePMCClient,
    find_sentences_with_substring,
    identify_tool_mentions_in_sentences,
    identify_tool_mentions_using_europepmc,
)


def test_search_mentions():
    client = EuropePMCClient()
    biotools_articles = client.search_mentions("ProteinProphet", article_limit=3)
    assert len(biotools_articles) == 3


def test_get_relevant_paragraphs():
    client = EuropePMCClient()
    tool_name = "ProteinProphet"
    biotools_articles = client.search_mentions(tool_name, article_limit=6)
    relevant_parahraphs = client.get_relevant_paragraphs(
        biotools_articles[0].pmcid, tool_name
    )
    assert len(relevant_parahraphs) == 8


def test_biotools():
    biotools = get_biotools("./biotoolspub/biotoolspub_with_topic.tsv")
    assert len(biotools) == 11151


def test_bio_tools_1():
    client = EuropePMCClient()
    biotools = get_biotools("./biotoolspub/biotoolspub_with_topic.tsv")
    biotools_articles = client.search_mentions(
        biotools[0].name, article_limit=1, topics=biotools[0].disjoint_topics()
    )


def test_identify_tool_mentions_using_europepmc():
    biotools = get_biotools("./biotoolspub/biotoolspub_with_topic.tsv")
    print(biotools[:1])
    tool_occurrences_df = identify_tool_mentions_using_europepmc(
        biotools[:1], article_limit=1, sentences_per_article=1
    )
    assert tool_occurrences_df.shape == (1, 4)


def test_topic_disjoint():
    biotools = get_biotools("./biotoolspub/biotoolspub_with_topic.tsv")
    tool = biotools[3]
    print(tool.name)
    client = EuropePMCClient()
    biotools_articles1 = client.search_mentions(
        tool.name, topics=tool.disjoint_topics()
    )
    biotools_articles2 = client.search_mentions(tool.name)
    assert len(biotools_articles1) < len(biotools_articles2)


def test_find_sentences_with_substring_basic():
    string_list = [
        "Hello world. This is a test sentence.",
        "Another test sentence here.",
        "The substring should be found here.",
    ]
    substring = "test"
    result = find_sentences_with_substring(string_list, substring)
    assert len(result) == 2


def test_find_sentences_with_substring_case_insensitive():
    string_list = [
        "Hello world. This is a Test sentence.",
        "Another TEST sentence here.",
        "The substring should not be found here.",
    ]
    substring = "test"
    result = find_sentences_with_substring(string_list, substring)
    assert len(result) == 2


def test_find_sentences_with_substring_no_match():
    string_list = [
        "Hello world. This is a test sentence.",
        "Another example sentence here.",
        "The substring should not be found here.",
    ]
    substring = "not_in_text"
    result = find_sentences_with_substring(string_list, substring)
    assert result == []


def test_find_sentences_with_substring_limit():
    string_list = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
        "Fourth test sentence.",
    ]
    substring = "test"
    result = find_sentences_with_substring(string_list, substring, limit=2)
    assert len(result) == 2


def test_find_sentences_with_substring_exceeding_limit():
    string_list = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
        "Fourth test sentence.",
    ]
    substring = "test"
    result = find_sentences_with_substring(string_list, substring, limit=10)
    assert len(result) == 4


def test_identify_tool_mentions_multiple_occurrences():
    tool = Tool_entry(
        name="Seurat",
        biotools_id="seurat",
        topics="RNA-Seq, Transcriptomics",
        pubmedid="34062119.0",
        pubmedcid="PMC8238499",
        link="fulltexts/PMC8238499.xml",
    )
    pmcid = "PMC8238499"
    paragraphs = [
        "Seurat is a popular tool for RNA-Seq analysis.",
        "The Seurat package provides various methods for transcriptomics data.",
        "No mention of the tool here.",
        "Seurat is widely used in the field of transcriptomics.",
    ]

    result = identify_tool_mentions_in_sentences(pmcid, tool, paragraphs, limit=3)

    # Check that the result contains 3 entries, due to the limit
    assert len(result) == 3
