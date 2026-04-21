import numpy as np
from transformers import AutoConfig  # ty: ignore[possibly-missing-import]


def retrieve_model_max_length(base_model) -> int:
    """Attempt to load the transformers configuration of a model and return the
    model max length"""
    try:
        model_max_length = AutoConfig.from_pretrained(
            base_model, trust_remote_code=True
        ).max_position_embeddings
    except Exception:
        model_max_length = np.nan
        raise ValueError(("Could not retrieve model's max length."))
    return model_max_length


def length_after_tokenizing(text: str, tokenizer):
    """Attempt at tokenizing a text and return the number of tokens"""
    try:
        return len(tokenizer(text).input_ids)
    except Exception:
        return np.nan
