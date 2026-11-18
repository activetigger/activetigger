from typing import Literal

TaskNames = Literal[
    "train_bert",
    "predict_bert",
    "train_image",
    "predict_image",
    "generation",
    "predict_with_features",
    "train_ner",
    "predict_ner",
    "extend_features",
    "compute_bert_embeddings",
    "compute_bertopic",
    "add_evalset",
    "compute_clip_imagexp",
    "compute_dfm",
    "compute_fasttext",
    "compute_lexicometrics",
    "compute_multimodal",
    "compute_projection",
    "compute_prompt_embedding",
    "compute_sbert",
    "compute_similarity_with_features",
    "create_project",
    "duplicate_project",
    "extend_features",
    "predict_ml",
    "prepare_dataset",
    "train_ml",  # was quickmodel
    "update_datasets",
]

# TODO: moves those in dedicated module?
features_tasks: list[TaskNames] = [
    "compute_sbert",
    "compute_bert_embeddings",
    "compute_clip_imagexp",
    "compute_multimodal",
    "compute_fasttext",
    "compute_dfm",
]

prompt_tasks: list[TaskNames] = ["compute_prompt_embedding", "compute_similarity_with_features"]

# Meta categories ?

#     "bertopic",
#     "lexicometrics","projection",
#     "predict_quickmodel",

#     "prompt_similarity",
#     "train_quickmodel",
