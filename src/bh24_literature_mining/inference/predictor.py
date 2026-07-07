import logging
import re

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, pipeline

from bh24_literature_mining.config import InferenceConfig

logger = logging.getLogger(__name__)

_PUNCT_ONLY = re.compile(r'^[\W\d_]+$')


def filter_predictions(preds: list[dict], min_len: int = 3) -> list[dict]:
    return [
        p for p in preds
        if len(p["word"].strip()) >= min_len and not _PUNCT_ONLY.match(p["word"].strip())
    ]


def truncate_if_needed(
    sentence: str, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> str:
    tokens = tokenizer(sentence, return_tensors="pt", truncation=False)
    if tokens["input_ids"].shape[1] > max_length:
        return tokenizer.decode(
            tokens["input_ids"][0][: max_length - 1], skip_special_tokens=True
        )
    return sentence


def format_entities(entity_list: list[dict], sentence: str) -> str:
    if not entity_list:
        return ""
    return "; ".join(
        f"{sentence[e['start']:e['end']]} ({e['entity_group']}) at {e['start']}-{e['end']}"
        for e in entity_list
    )


def predict_batch(
    sentences: list[str],
    classifier: object,
    batch_size: int = 16,
) -> list[list[dict]]:
    dataset = Dataset.from_dict({"text": sentences})
    results: list[list[dict]] = []
    for out in classifier(dataset["text"], batch_size=batch_size):
        results.append(out if isinstance(out, list) else [out])
    return results


def run_inference(
    config: InferenceConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_position_embeddings: int = 512,
) -> pd.DataFrame:
    classifier = pipeline(
        "ner",
        model=str(config.checkpoint_path),
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    to_predict_df = pd.read_csv(config.input_path)
    sentences = [
        truncate_if_needed(s, tokenizer, max_position_embeddings)
        for s in to_predict_df["Sentence"]
    ]

    results = predict_batch(sentences, classifier, config.batch_size)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(config.output_path, index=False)

    to_predict_df["NER_Model_Found"] = [
        format_entities(r, s) for r, s in zip(results, sentences)
    ]
    return to_predict_df
