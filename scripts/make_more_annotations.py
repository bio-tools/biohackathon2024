#!/usr/bin/env python3
import logging
from datetime import datetime
from pathlib import Path

from bh24_literature_mining.biotools import get_biotools
from bh24_literature_mining.europepmc_api import identify_tool_mentions_using_europepmc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
p_out = PROJECT_ROOT / "data" / "annotated"
p_out.mkdir(parents=True, exist_ok=True)

biotools = get_biotools(
    str(PROJECT_ROOT / "biotoolspub" / "biotoolspub_with_topic.tsv"), limit=99999
)

tool_occurrences_df = identify_tool_mentions_using_europepmc(
    biotools[4000:6000], article_limit=3
)

current_date = datetime.now().strftime("%y%m%d")
out_path = p_out / f"{current_date}_mentions_with_topics.csv"
tool_occurrences_df.to_csv(out_path, index=False)
logging.getLogger(__name__).info(
    "Wrote %d rows to %s", len(tool_occurrences_df), out_path
)
