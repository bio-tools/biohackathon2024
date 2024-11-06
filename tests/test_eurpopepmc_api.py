
from bh24_literature_mining.europepmc_api import EuropePMCClient


def test_search_mentions():
    client = EuropePMCClient()
    biotools_articles = client.search_mentions("ProteinProphet", article_limit = 3)
    assert len(biotools_articles) == 3

def test_get_relevant_paragraphs():
    client = EuropePMCClient()
    tool_name = "ProteinProphet"
    biotools_articles = client.search_mentions(tool_name, article_limit = 3)

    relevant_parahraphs = client.get_relevant_paragraphs(biotools_articles[0].pmcid, tool_name)
    assert len(relevant_parahraphs) == 8