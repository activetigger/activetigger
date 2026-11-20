import logging

import numpy as np
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,  # ty: ignore[possibly-missing-import]
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
    PreTrainedTokenizerFast,  # ty: ignore[possibly-missing-import]
)
from transformers.utils import cached_file

TOKENIZER_PROBE = (
    "L'été dernier, le gouvernement a annoncé 3 réformes particulièrement contestées — isn't it?"
)


def load_tokenizer(name_or_path, **kwargs):
    """
    AutoTokenizer checked against the repo's tokenizer.json, 
    with fallback on mismatch, solve a regression problem
    """
    name_or_path = str(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    try:
        path = cached_file(
            name_or_path,
            "tokenizer.json",
            local_files_only=kwargs.get("local_files_only", False),
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if path is None:
            return tokenizer
        reference = Tokenizer.from_file(path)
        reference.no_padding()
        reference.no_truncation()
        expected = reference.encode(TOKENIZER_PROBE).ids
    except Exception:
        return tokenizer
    if tokenizer(TOKENIZER_PROBE)["input_ids"] == expected:
        return tokenizer
    # declared tokenizer class does not match tokenizer.json (e.g. camembertav2)
    logging.warning(f"Tokenizer class mismatch for {name_or_path}, loading tokenizer.json as is")
    fallback = PreTrainedTokenizerFast.from_pretrained(name_or_path, **kwargs)
    if fallback(TOKENIZER_PROBE)["input_ids"] != expected:
        raise ValueError(f"Tokenizer of {name_or_path} is inconsistent with its tokenizer.json")
    return fallback


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
