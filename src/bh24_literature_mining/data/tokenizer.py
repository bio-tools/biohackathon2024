from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
