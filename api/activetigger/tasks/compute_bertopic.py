import datetime
import gc
import json
import shutil
from pathlib import Path
from string import punctuation

import hdbscan  # type: ignore[import]
import nltk  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import plotly.graph_objects as go  # type: ignore[import]
import torch
import umap  # type: ignore[import]
from bertopic import BERTopic  # type: ignore[import]
from great_tables import GT, loc, style
from jinja2 import Template
from nltk.corpus import stopwords as nltk_stopwords  # type: ignore[import]
from simplemma import lemmatize  # type: ignore[import]
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import]
from slugify import slugify  # type: ignore[import]

from activetigger.config import config
from activetigger.datamodels import BertopicParamsModel, EventsModel
from activetigger.monitoring import TaskTimer
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.compute_sbert import ComputeSbert

nltk.download("stopwords", quiet=True)

# accelerate UMAP
try:
    import cuml  # type: ignore[import-not-found]

    CUMl_AVAILABLE = True
except ImportError:
    print("CuML not installed")
    CUMl_AVAILABLE = False

"""
Rational : 
- Get texts / existing embeddings / parameters
- Compute embeddings if not provided
- Compute the projection in 2D
- Compute the BERTOPIC pipeline
- Save the results to display
"""

# TODO : multicolumns for text
# TODO : manage special case of embeddings of trainset


ISO_TO_NLTK = {
    "ar": "arabic",
    "az": "azerbaijani",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "kk": "kazakh",
    "ne": "nepali",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tg": "tajik",
    "tr": "turkish",
}


def visualize_documents(
    topics: list[int],
    topic_info: pd.DataFrame,
    embeddings: list[list[float]] | np.ndarray,
    docs: list[str] | np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    min_number_of_element: int = 50,
    random_seed: int = 42,
) -> go.Figure:
    """"""
    # Transform inputs in np.ndarray
    embeddings = np.array(embeddings)
    topics = np.array(topics)  # type: ignore[assignment]

    # Reduce embeddings
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_seed,
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    X, Y = reduced_embeddings[:, 0], reduced_embeddings[:, 1]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X,
                y=Y,
                mode="markers",
                hoverinfo="skip",
                marker={"color": "#bababa", "opacity": 0.3},
                showlegend=False,
            )
        ],
        layout={
            "width": 900,
            "plot_bgcolor": "#FCFCFC",
            "margin": {"t": 20, "b": 40},
            "xaxis": {"zerolinecolor": "#ECECEC", "gridcolor": "#ECECEC"},
            "yaxis": {"zerolinecolor": "#ECECEC", "gridcolor": "#ECECEC"},
        },
    )

    def adapt_length_to_width(text: str, width: int = 100, max_n_words: int = 10):
        output = ""
        len_line = 0
        n_words = 0
        for word in text.split(" "):
            if len(word) + len_line > width:
                output += "<br>" + word + " "
                len_line = len(word) + 1
            else:
                output += word + " "
                len_line += len(word) + 1

            n_words += 1
            if n_words > max_n_words:
                break
        return output

    # def hovertext_transform(docs: list[str] | np.ndarray, width: int = 100, max_n_words: int = 100):
    #     """"""
    #     return np.array([adapt_length_to_width(doc, width, max_n_words) for doc in docs])

    # hovertext = hovertext_transform(docs)

    for topic in np.unique(topics):
        if topic == -1:
            continue
        indexes: np.ndarray = topics == topic
        if sum(indexes) < min_number_of_element:
            continue
        fig.add_trace(
            go.Scatter(
                x=X[indexes],
                y=Y[indexes],
                mode="markers",
                # hovertemplate = "%{text}",
                # text = hovertext[indexes],
                marker={"opacity": 0.75},
                name=topic_info.loc[topic == topic_info.Topic, "Name"].item(),
            )
        )
    return fig


class CustomLemmatizer:
    """An object to apply the lemmatize function."""

    def __init__(self, language, stop_words) -> None:
        self.__language = language
        self.__stop_words = stop_words

    def __call__(self, doc: str) -> list[str]:
        """ """
        doc = doc.lower()
        doc = "".join([c for c in doc if c not in punctuation])
        out = []
        for word in doc.split(" "):
            if (word not in self.__stop_words) and (len(word) > 0):
                try:
                    lemma = lemmatize(word, lang=self.__language)
                except Exception as e:
                    # if lemmatization failed, skip it and use the word instead
                    lemma = word
                    print(f"BERTopic - Lemmatization failed (word: {word}) - Error : {e}")
                if lemma not in self.__stop_words:
                    out += [lemma]
        return out


class ComputeBertopic(BaseTask):
    """
    Compute BERTopic model

    A computation is identitied by its name that should be unique
    and is associated with parameters

    Embeddings are computed if not provided or if force_compute_embeddings is True.
    Embeddings are identified by run name + embedding model name.
    """

    kind = "compute_bertopic"

    def __init__(
        self,
        path_bertopic: Path,
        path_data: Path,
        col_id: str | None,
        col_text: str,
        parameters: BertopicParamsModel,
        name: str | None = None,
        existing_embeddings: Path | None = None,
        cols_embeddings: list[str] | None = None,
        force_compute_embeddings: bool = False,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        if existing_embeddings and not existing_embeddings.suffix == ".parquet":
            raise ValueError("Embeddings file must be a parquet file.")

        # Set parameters
        self.random_seed = random_seed
        self.path_data = path_data
        self.col_id = col_id
        self.col_text = col_text
        if name is None:
            name = f"bertopic_{path_data.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.name = slugify(name)
        self.input_datasets = (
            parameters.input_datasets
        )  # train, all_sets (ie train+valid+test), complete
        self.parameters = parameters
        self.existing_embeddings = (
            existing_embeddings  # Path to force using one file for the embeddings
        )
        self.cols_embeddings = cols_embeddings
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.force_compute_embeddings = force_compute_embeddings

        # Create paths
        self.path_bertopic = path_bertopic
        self.path_bertopic.joinpath("embeddings").mkdir(parents=True, exist_ok=True)

        # Check if the run already exists
        self.path_run = self.path_bertopic.joinpath("runs").joinpath(self.name)
        if self.path_run.exists():
            raise ValueError(f"Run {self.name} already exists, please chose another name.")

    def __stop_process_opportunity(self):
        """stop process by the user"""
        if self.event is not None:
            if self.event.is_set():
                raise Exception("Process interrupted by user")

    def __init_paths(self) -> tuple[Path, Path]:
        """Creates a folder (projects/{project_slug}/bertopic/runs/{bertopic_run}/)
        as well a path for the embeddings (common to multiple runs —
        projects/{project_slug}/bertopic/embeddings/...) and a path for the projection
        (unique per run — projects/{project_slug}/bertopic/runs/{bertopic_run}/projection2D.parquet)"""

        # Create directory for the run
        self.path_run.mkdir(parents=True, exist_ok=True)
        path_embeddings = self.path_bertopic.joinpath("embeddings").joinpath(
            (
                f"bertopic_embeddings_{self.input_datasets}_"
                f"{slugify(self.parameters.embedding_model)}"
                f".parquet"
            )
        )
        path_projection = self.path_run.joinpath("projection2D.parquet")
        return path_embeddings, path_projection

    def __load_data(self) -> pd.DataFrame:
        """Depending on the input_datasets (train, all_sets or complete), load
        one or several files and return it"""

        match self.input_datasets:
            case "train":
                df = pd.read_parquet(self.path_data.joinpath(config.train_file))
            case "all_sets":
                df = pd.concat(
                    [
                        pd.read_parquet(self.path_data.joinpath(file))
                        for file in [config.train_file, config.test_file, config.valid_file]
                        if self.path_data.joinpath(file).is_file()
                    ]
                )
            case "complete":
                df = pd.read_parquet(self.path_data.joinpath(config.data_all))
        return df

    def __check_text_data(self, df) -> pd.DataFrame:
        """Check the validity of the dataframe (contains necessary data, index,
        and text length) and raise errors if necessary"""

        if self.col_text not in df.columns:
            raise ValueError(f"Column {self.col_text} not found in the data.")
        # Set the index if col_id is provided
        if self.col_id and self.col_id in df.columns:
            df.set_index(self.col_id, inplace=True)

        # Drop rows with a text length too small
        criterion = df[self.col_text].apply(len) > self.parameters.filter_text_length
        df = df[criterion]
        print(f"Data loaded with {len(df)} rows.")

        if len(df) < 15:
            raise ValueError(
                f"Not enough elements ({len(df)}) — after "
                f"removing elements < {self.parameters.filter_text_length}"
            )
        return df

    def __check_if_embeddings_computation_necessary(self, path_embeddings: Path) -> bool:
        """Check if embeddings must be computed or not.
        We compute embeddings if:
            - force_compute_embeddings is True
            - existing_embeddings is not provided and the embeddings were not
                previously computed.
        TODO: AM: Might need to clarify which of force_compute_embeddings or
        existing_embeddings has the priority
        """
        return (
            not self.existing_embeddings and not path_embeddings.exists()
        ) or self.force_compute_embeddings

    def __compute_embeddings(self, df: pd.DataFrame, path_embeddings: Path) -> None:
        """
        Compute the embeddings using the SBERT model
        """
        if self.parameters.embedding_kind != "sentence_transformers":
            raise ValueError("Only sentence_transformers embeddings are supported for BERTopic.")
        if path_embeddings.exists():
            path_embeddings.unlink()

        if self.parameters.embedding_model not in config.models_embeddings:
            raise ValueError(f"Model {self.parameters.embedding_model} not supported.")
        self.update_progress(f"Computing embeddings with {self.parameters.embedding_model}")
        embeddings = ComputeSbert(
            texts=df[self.col_text],
            path_process=self.path_bertopic,
            model=self.parameters.embedding_model,
            batch_size=32,
            min_gpu=1,
            path_progress=self.path_run.joinpath("progress"),
        )
        # transmit the event to allow interruption
        embeddings.event = self.event
        # launch computation
        computed = embeddings()
        # save the embeddings to a file
        computed.to_parquet(path_embeddings)

    def __load_embeddings(self, path_embeddings: Path) -> pd.DataFrame:
        """Load the embeddings, No verification because the file should exist,
        errors are flagged beforehand"""
        return pd.read_parquet(path_embeddings)

    def __check_embeddings(
        self, df_embeddings: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Retrieve the embeddings for the each text element in the dataframe. If
        some are missing, restart embedding computation.
        Make sure that the embeddings loaded have the right form and that the index
        of the embeddings_df and the df (text) match
        """
        # Check if all text have it's embedding
        if False in np.isin(df.index, df_embeddings.index):
            print("Some elements did not have their embedding")
            return None

        try:
            df_embeddings = df_embeddings.loc[df.index, :]
        except Exception as e:
            raise ValueError(
                "The index of the embeddings and the texts do not match. "
                "Please force the computation of embeddings or check your data."
                "\n"
                "Python error: {e}"
            )
        if self.cols_embeddings:
            # NOTE: AM: Artefact?
            if not all(col in df_embeddings.columns for col in self.cols_embeddings):
                raise ValueError("Some embeddings columns not found in the embeddings data.")
            df_embeddings = df_embeddings[self.cols_embeddings]

        return df_embeddings

    def __create_projection(self, df_embeddings: pd.DataFrame, path_projection: Path) -> None:
        """
        Compute the projection to display it in the frontend later on
        """
        try:
            reducer = cuml.UMAP(
                n_neighbors=self.parameters.umap_n_neighbors,
                n_components=2,
                min_dist=0.0,
                metric="cosine",
                random_state=self.random_seed,  # for deterministic behaviour
            )
            print("Using cuML for UMAP computation")
        except Exception:
            reducer = umap.UMAP(
                n_neighbors=self.parameters.umap_n_neighbors,
                n_components=2,
                min_dist=0.0,
                low_memory=False,
                metric="cosine",
                random_state=self.random_seed,  # for deterministic behaviour
            )
            print("Using standard UMAP for computation")
        reduced_embeddings: np.ndarray = reducer.fit_transform(df_embeddings.values)
        del reducer
        df_reduced = pd.DataFrame(reduced_embeddings, index=df_embeddings.index, columns=["x", "y"])
        df_reduced.to_parquet(path_projection)

    def __load_UMAP_HDBSCAN(self):
        # Dimensionality reduction with UMAP
        try:
            umap_model = cuml.UMAP(
                n_neighbors=self.parameters.umap_n_neighbors,
                n_components=self.parameters.umap_n_components,
                # min_dist=self.parameters.umap_min_dist, # Removed because 0.0 is the best value to use for clustering - Axel
                min_dist=0.0,
                metric="cosine",
                random_state=self.random_seed,  # for deterministic behaviour
            )
        except Exception as e:
            print(f"CuML UMAP failed: {e}, using standard UMAP instead.")
            umap_model = umap.UMAP(
                n_neighbors=self.parameters.umap_n_neighbors,
                n_components=self.parameters.umap_n_components,
                # min_dist=self.parameters.umap_min_dist, # Removed because 0.0 is the best value to use for clustering - Axel
                min_dist=0.0,
                metric="cosine",
                low_memory=False,
                random_state=self.random_seed,  # for deterministic behaviour
            )

        # Clustering with HDBSCAN
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.parameters.hdbscan_min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        )
        return umap_model, hdbscan_model

    def __load_vectorizer(self) -> CountVectorizer:
        """Load a Vectorizer model, tries to load a custom lemmatizer if, failed
        return a default CountVectorizer.

        TODO: AM: user should be warned if the vectorizer model is a default
            CountVectorizer
        """
        try:
            lang = self.parameters.language
            if lang not in ISO_TO_NLTK:
                raise ValueError(f"Unsupported language {self.parameters.language}")

            stopwords = list(nltk_stopwords.words(ISO_TO_NLTK[lang]))
            vectorizer_model = CountVectorizer(tokenizer=CustomLemmatizer(lang, stopwords))
        except ValueError:
            vectorizer_model = CountVectorizer()
        return vectorizer_model

    def __create_saving_files(
        self,
        df: pd.DataFrame,
        topic_model: BERTopic,
        topics: list[int],
        embeddings: np.ndarray,
        path_embeddings: Path,
        path_projection: Path,
    ) -> None:
        """Create the following files:
        - topics csv (topic info)
        - clusters csv (list binding id - cluster)
        - params.json
        - report.html (through __create_report)
        """
        # Add the topics to the DataFrame
        df["cluster"] = topics

        # Save the topics and documents informations
        topics_df: pd.DataFrame = topic_model.get_topic_info()
        if self.parameters.outlier_reduction:
            topics_df = topics_df.loc[topics_df.Topic != -1, :]
            # Update the counts
            topics_df = topics_df.set_index("Topic")
            topics_df["Count-Bis"] = pd.Series(topics).value_counts()
            topics_df = topics_df.reset_index()

        topics_df.to_csv(self.path_run.joinpath("bertopic_topics.csv"), index=False)
        (
            df.reset_index()[  # id_internal is the index, to make it clearer, i reset the index and select id_internal in the cols to save
                ["id_internal", "id_external", "cluster"]
            ].to_csv(self.path_run.joinpath("bertopic_clusters.csv"), index=False)  # cols to save
        )

        with open(self.path_run.joinpath("params.json"), "w") as f:
            json.dump(
                {
                    "bertopic_params": self.parameters.model_dump(),
                    "col_text": self.col_text,
                    "col_id": self.col_id,
                    "name": self.name,
                    "timestamp": self.timestamp,
                    "path_data": str(self.path_data),
                    "path_embeddings": str(path_embeddings),
                    "path_projection": str(path_projection),
                },
                f,
            )
        # Create a report
        self.__create_report(
            topic_model=topic_model,
            topics=topics,
            topic_info=topics_df,
            docs=df[self.col_text].to_list(),
            embeddings=embeddings,
        )

    def __create_report(
        self,
        topic_model: BERTopic,
        topics: list[int],
        topic_info: pd.DataFrame,
        docs: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """Creates an HTML report downloadable by the user"""
        # Creates a table for topic info with great tables
        topic_info = topic_info.copy()
        topic_info["Representation"] = topic_info["Representation"].apply(lambda l: " - ".join(l))
        table_topics = (
            GT(topic_info.drop(columns=["Name", "Representative_Docs"]))
            .cols_align("center", "Count")
            .tab_header(title="Topic info", subtitle=None)
            .cols_width(cases={"Topic": "10%", "Count": "10%", "Representation": "60%"})
            .tab_style(
                style=[style.text(align="right"), style.borders(sides="right", color="#d3d3d3")],
                locations=loc.body(columns="Topic"),
            )
        )

        # Create a table for the topic model settings
        settings = {
            "language": self.parameters.language,
            "top_n_words": self.parameters.top_n_words,
            "n_gram_range": self.parameters.n_gram_range,
            "outlier_reduction": self.parameters.outlier_reduction,
            "hdbscan_min_cluster_size": self.parameters.hdbscan_min_cluster_size,
            "umap_n_neighbors": self.parameters.umap_n_neighbors,
            "umap_n_components": self.parameters.umap_n_components,
            "embedding_kind": self.parameters.embedding_kind,
            "embedding_model": self.parameters.embedding_model,
            "filter_text_length": self.parameters.filter_text_length,
        }
        setting_df = pd.DataFrame({"Parameter": settings.keys(), "Value": settings.values()})

        table_settings = (
            GT(setting_df)
            .cols_align("right", "Parameter")
            .cols_align("right", "Value")
            .cols_width(
                cases={
                    "Parameter": "40%",
                    "Value": "40%",
                }
            )
        )

        # Create Plotly figure for 2D maps
        try:
            fig_map = visualize_documents(
                topics=topics,
                topic_info=topic_info,
                docs=docs,
                embeddings=embeddings,
                n_neighbors=self.parameters.umap_n_neighbors,
                min_dist=0.0,
                min_number_of_element=-1,  # Need additional implementation
                random_seed=self.random_seed,
            )
        except:
            fig_map = "You don't have enough elements to compute the hierarchy visualisation"

        # Create plotly figure for hierarchical representation
        fig_hierarchical = topic_model.visualize_hierarchy().update_layout(
            width=900,
            title={"text": ""},
            margin={"t": 20, "b": 40},
            plot_bgcolor="#FCFCFC",
        )
        # Export results
        saving_kwargs = {
            "full_html": False,
            "include_plotlyjs": "cdn",
            "include_mathjax": "cdn",
            "config": {
                "responsive": False,
                "modeBarButtonsToRemove": ["zoomIn", "zoomOut", "autoScale", "select"],
                "displaylogo": False,
                "displayModeBar": False,
            },
        }

        jinja_data = {
            "bertopic_name": self.name.replace("-", " "),
            "topics": table_topics.as_raw_html(),
            "map": fig_map.to_html(
                **saving_kwargs,
            ),
            "hierarchical": fig_hierarchical.to_html(
                **saving_kwargs,
            ),
            "RepresentativeTopics": {
                topic_repr: topic_docs
                for topic_repr, topic_docs in zip(
                    topic_info.Representation, topic_info.Representative_Docs
                )
            },
            "parameters": table_settings.as_raw_html(),
        }

        input_template_path = "./activetigger/html/bertopic_report_template.html"
        output_file_path = self.path_run.joinpath("report.html")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            with open(input_template_path) as template_file:
                j2_template = Template(template_file.read())
                output_file.write(j2_template.render(jinja_data))

    def __call__(self) -> EventsModel:
        """
        Compute BERTopic model
        Main steps:
            - create folder
            - load files
            - (if necessary compute the embeddings &) Load embeddings
            - Create BERTopic instance + fit_transform
            - Save parameters, results and a n html report
        """
        task_timer = TaskTimer(
            compulsory_steps=["setup", "fit", "save_files"],
            optional_steps=["reduce-outliers", "generate-embeddings"],
        )
        try:
            task_timer.start("setup")
            path_embeddings, path_projection = self.__init_paths()
            self.update_progress("Initializing")

            df = self.__load_data()
            df = self.__check_text_data(df)

            if self.__check_if_embeddings_computation_necessary(path_embeddings):
                task_timer.start("generate-embeddings")
                self.__compute_embeddings(df, path_embeddings)
                task_timer.stop("generate-embeddings")

            df_embeddings = self.__load_embeddings(path_embeddings)
            df_embeddings_t = self.__check_embeddings(df_embeddings, df)

            if df_embeddings_t is None:
                # If issue when checking the embeddings, force compute the embeddings
                task_timer.start("generate-embeddings")
                self.__compute_embeddings(df, path_embeddings)
                task_timer.stop("generate-embeddings")
                df_embeddings = self.__load_embeddings(path_embeddings)
                df_embeddings_t = self.__check_embeddings(df_embeddings, df)

            if df_embeddings_t is None:
                raise Exception(
                    "Something went wrong when checking the embeddings. Please check the logs."
                )

            embeddings: np.ndarray = df_embeddings_t.values
            self.__create_projection(df_embeddings_t, path_projection)

            self.__stop_process_opportunity()

            # Initialize BERTopic
            self.update_progress("Initializing BERTopic")
            umap_model, hdbscan_model = self.__load_UMAP_HDBSCAN()
            vectorizer_model = self.__load_vectorizer()

            topic_model = BERTopic(
                language=self.parameters.language,
                vectorizer_model=vectorizer_model,
                # nr_topics=self.parameters.nr_topics, # Removed because overridden by the hdbscan model - Axel
                # min_topic_size=self.parameters.min_topic_size, # Removed to propose topic reduction later in the pipeline - Axel
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
            )
            task_timer.stop("setup")

            task_timer.start("fit")
            self.update_progress(f"Fitting the model on {embeddings.shape[0]} elements")
            print(f"Fitting the model on {embeddings.shape} / {len(df)} elements")
            # Fit the BERTopic model
            topics, _ = topic_model.fit_transform(
                documents=df[self.col_text],
                embeddings=embeddings,
            )
            task_timer.stop("fit")

            # Add outlier reduction
            if self.parameters.outlier_reduction:
                task_timer.start("reduce-outliers")
                try:
                    print("Reducing outliers")
                    topics = topic_model.reduce_outliers(
                        documents=df[self.col_text],
                        topics=topics,
                        embeddings=embeddings,
                        strategy="embeddings",
                    )
                except Exception as e:
                    self.parameters.outlier_reduction = False
                    print(
                        f"Error during outlier reduction: {e} — Most likely "
                        "because there are not enough elements in your dataset"
                    )
                task_timer.stop("reduce-outliers")

            self.__stop_process_opportunity()

            task_timer.start("save_files")
            self.__create_saving_files(
                df=df,
                topic_model=topic_model,
                topics=topics,
                embeddings=embeddings,
                path_projection=path_projection,
                path_embeddings=path_embeddings,
            )
            task_timer.stop("save_files")

            self.path_run.joinpath("progress").unlink(missing_ok=True)
            return EventsModel(events=task_timer.get_events())

        except Exception as e:
            # Case an error happens
            if self.path_run.exists():
                shutil.rmtree(self.path_run)
            if "Found array with 0 sample(s)".lower() in str(e).lower():
                # TODO Make it nicer with proper notification center
                e = Exception(
                    (
                        f"{str(e)}"
                        "\n"
                        f"[Found array with 0 sample(s)] errors are likely due to"
                        f"your dataset being to small ({len(df)}). We advise you "
                        f"to add more elements to your dataset."
                    )
                )

            raise e
        finally:
            print("Cleaning GPU memory (bertopic)")
            try:
                del (
                    df,
                    df_embeddings,
                    embeddings,
                    topic_model,
                    umap_model,
                    hdbscan_model,
                    vectorizer_model,
                    self.event,
                )
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def update_progress(self, message: str) -> None:
        """
        Update the progress of the task
        """
        with open(self.path_run.joinpath("progress"), "w") as f:
            f.write(message)
