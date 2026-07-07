from transformers import AutoModelForTokenClassification

from bh24_literature_mining.config import ModelConfig


def create_model(
    config: ModelConfig,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> AutoModelForTokenClassification:
    return AutoModelForTokenClassification.from_pretrained(
        config.pretrained,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        ignore_mismatched_sizes=True,
    )
