import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


def find_sub_span(
    token_span: tuple[int, int], entity_span: tuple[int, int]
) -> tuple[int, int] | None:
    if token_span[0] < entity_span[1] and token_span[1] > entity_span[0]:
        return max(token_span[0], entity_span[0]), min(token_span[1], entity_span[1])
    return None


def _tokenize_to_words(
    text: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[str], list[tuple[int, int]]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    wids = enc.word_ids()
    words: list[str] = []
    word_spans: list[tuple[int, int]] = []
    prev_wid = None
    for idx, wid in enumerate(wids):
        if wid is None:
            continue
        if wid != prev_wid:
            words.append(text[offsets[idx][0] : offsets[idx][1]])
            word_spans.append((offsets[idx][0], offsets[idx][1]))
        else:
            start = word_spans[-1][0]
            end = offsets[idx][1]
            word_spans[-1] = (start, end)
            words[-1] = text[start:end]
        prev_wid = wid
    return words, word_spans


def convert_to_iob(
    texts: list[str],
    ner_tags_list: list,
    tokenizer: PreTrainedTokenizerBase,
) -> list[list[tuple[str, str]]]:
    results = []
    for text, ner_tags in zip(texts, ner_tags_list):
        if not text:
            results.append([])
            continue

        words, word_spans = _tokenize_to_words(text, tokenizer)

        if ner_tags is None:
            results.append(list(zip(words, ["O"] * len(words))))
            continue

        iob_tags = ["O"] * len(words)
        for start, end, entity, entity_type in sorted(ner_tags, key=lambda x: x[0]):
            entity_flag = False
            for i, word_span in enumerate(word_spans):
                if find_sub_span(word_span, (start, end)):
                    if not entity_flag:
                        iob_tags[i] = "B-" + entity_type
                        entity_flag = True
                    elif iob_tags[i] == "O":
                        iob_tags[i] = "I-" + entity_type
                else:
                    entity_flag = False

        results.append(list(zip(words, iob_tags)))
    return results


def convert_to_IOB_format_from_df(
    dataframe: pd.DataFrame,
    output_folder: Path,
    filename: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 500,
) -> None:
    data = [(row["Sentence"], row["NER_Tags"]) for _, row in dataframe.iterrows()]
    result_path = output_folder / filename
    with open(result_path, "w", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i : i + batch_size]
            sentences, ner_tags_batch = zip(*batch)
            batch_results = convert_to_iob(
                list(sentences), list(ner_tags_batch), tokenizer
            )
            for tagged_tokens in batch_results:
                for each_token in tagged_tokens:
                    writer.writerow(list(each_token))
                writer.writerow("")


def load_iob_file(file_path: Path) -> pd.DataFrame:
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    data.append((parts[0], parts[1]))
            else:
                data.append(("", ""))
    return pd.DataFrame(data, columns=["token", "tag"])
