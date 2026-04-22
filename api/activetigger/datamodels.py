import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional

from pandas import DataFrame
from pydantic import BaseModel, BeforeValidator, ConfigDict  # for dataframe
from sklearn.base import BaseEstimator

# Data model to use of the API


class ChangePasswordModel(BaseModel):
    """
    Model for changing password
    """

    pwdold: str
    pwd1: str
    pwd2: str


class PredictedLabel(BaseModel):
    label: str | None
    proba: float | None
    entropy: float | None = None


class QueueTaskModel(BaseModel):
    """
    Task in the queue
    """

    unique_id: str
    kind: str
    project_slug: str
    state: str
    future: Optional[Any] = None  # Future object from concurrent.futures
    event: Any = None  # Event object for signaling
    starting_time: datetime.datetime
    queue: str
    task: Callable[..., Any] | None


class QueueStateTaskModel(BaseModel):
    """
    Task in the queue with state
    """

    unique_id: str
    kind: str
    state: str
    exception: Any = None


class ProjectBaseModel(BaseModel):
    """
    Parameters of a project to save in the database
    """

    cols_text: list[str]
    project_name: str
    col_id: str
    n_train: int
    n_test: int
    n_valid: int = 0
    from_project: str | None = None
    from_toy_dataset: bool = False
    filename: str | None = None
    dir: Path | None = None
    embeddings: list[str] = []
    n_skip: int = 0
    default_scheme: list[str] = []
    language: str = "fr"
    cols_label: list[str] = []
    cols_context: list[str] = []
    test: bool = False
    valid: bool = False
    n_total: int | None = None
    clear_test: bool = False
    clear_valid: bool = False
    train_selection:Literal["sequential","stratification","force_label","random"]="random"
    holdout_selection:Literal["sequential","random","stratification",None]=None
    s_val_idx:int|None=None
    s_test_idx:int|None=None
    cols_stratify: list[str] = []
    force_computation: bool = False
    seed: int = 42


class ProjectModel(ProjectBaseModel):
    """
    Once created
    """

    project_slug: str
    all_columns: list[str] | None = None


class ProjectDataModel(ProjectBaseModel):
    """
    To create a new project
    """

    csv: str


class AnnotationsDataModel(BaseModel):
    col_id: str
    col_label: str
    scheme: str
    csv: str
    filename: str | None = None


class EvalSetDataModel(BaseModel):
    cols_text: list[str]
    col_id: str
    n_eval: int
    filename: str
    csv: str
    col_label: str | None = None
    scheme: str | None = None


class ActionModel(str, Enum):
    """
    Type of actions available
    """

    delete = "delete"
    add = "add"
    update = "update"


class ActiveModel(BaseModel):
    """
    Active learning model
    """

    type: str
    value: str
    label: str
    time: str = ""
    labels_excluded: list[str] = []


class NextInModel(BaseModel):
    """
    Requesting next element to annotate
    """

    scheme: str
    selection: str = "fixed"
    sample: str = "untagged"
    on_labels: list[str] | None = None
    on_users: list[str] | None = None
    label_prob: str | None = None
    frame: list[Any] | None = None
    history: list[str] = []
    filter: str | None = None
    dataset: str = "train"
    model_active: ActiveModel | None = None


class ElementInModel(BaseModel):
    """
    Requesting element to annotate
    """

    element_id: str
    dataset: str
    scheme: str | None = None
    active_model: ActiveModel | None = None


class AnnotationModel(BaseModel):
    """
    Complete information on an annotation
    """

    project_slug: str
    dataset: str
    scheme: str
    element_id: str
    label: str | None = None
    time: datetime.datetime | None = None
    user: str | None = None
    comment: str | None = None
    selection: str | None = None


class ElementOutModel(BaseModel):
    """
    Posting element to annotate
    """

    element_id: str
    text: str
    context: dict[str, Any]
    selection: str
    info: str | None
    predict: PredictedLabel
    frame: list | None
    limit: int | None  # TO REMOVE
    history: list[AnnotationModel] | None = None
    n_sample: int | None = None


class NewUserModel(BaseModel):
    """
    New user definition
    """

    username: str
    password: str
    contact: str
    status: str


class UserModel(BaseModel):
    """
    User definition
    """

    username: str
    status: str | None = None
    contact: str | None = None


class UserInDBModel(UserModel):
    """
    Adding password to user definition
    """

    hashed_password: str


class CompareSchemesModel(BaseModel):
    """
    Compare two schemes
    """

    datetime: datetime.datetime
    project_slug: str
    schemeA: str
    schemeB: str
    labels_overlapping: float
    n_annotated: int | None = None
    cohen_kappa: float | None = None
    percentage: float | None = None


class TokenModel(BaseModel):
    """
    Auth token
    """

    access_token: str
    token_type: str
    status: str | None


class TableAnnotationsModel(BaseModel):
    """
    Table of annotations
    """

    annotations: list[AnnotationModel]
    dataset: str | None = "train"


class SchemeModel(BaseModel):
    """
    Specific scheme
    """

    project_slug: str
    name: str
    kind: str = "multiclass"
    labels: list[Annotated[str, BeforeValidator(lambda v: str(v))]] = []


class RegexModel(BaseModel):
    """
    Regex
    """

    project_slug: str
    name: str
    value: str
    user: str
    regex_count: bool = False


class LMParametersModel(BaseModel):
    """
    Parameters for bertmodel training
    """

    batchsize: int = 4
    gradacc: float = 1
    epochs: int = 3
    lrate: float = 5e-05
    wdecay: float = 0.01
    best: bool = True
    eval: int = 10
    gpu: bool = False
    adapt: bool = True


class LMParametersModelTrained(LMParametersModel):
    """
    Parameters for bertmodel once trained
    """

    base_model: str
    n_train: int
    test_size: float


class LMParametersDbModel(LMParametersModel):
    predicted: bool = False
    compressed: bool = False


class LMStatusModel(BaseModel):
    predicted: bool = False
    predicted_all: bool = False
    tested: bool = False
    predicted_external: bool = False
    name: str
    time: str


class BertModelModel(BaseModel):
    """
    Request Bertmodel
    TODO : model for parameters
    """

    project_slug: str
    scheme: str
    name: str
    base_model: str
    params: LMParametersModel
    test_size: float = 0.2
    dichotomize: str | None = None
    class_min_freq: int = 1
    class_balance: bool = False
    loss: str = "cross_entropy"
    exclude_labels: list[str] = []
    max_length: int = 512
    auto_max_length: bool = False


class UmapModel(BaseModel):
    """
    Params UmapModel
    """

    n_neighbors: int
    n_components: int
    min_dist: float


class TsneModel(BaseModel):
    """
    Params TsneModel
    """

    n_components: int
    learning_rate: str | float
    init: str
    perplexity: int


class ProjectionParametersModel(BaseModel):
    """
    Request projection
    """

    method: str
    features: list
    parameters: dict[str, float | str | bool | list] = {}
    normalize_features: bool = False


class ProjectionDataModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    data: DataFrame
    parameters: ProjectionParametersModel


class ProjectionOutModelNode(BaseModel):
    node_id: str
    label: str
    x: float
    y: float
    predictions: list | None = None


class ProjectionOutModel(BaseModel):
    """
    Posting projection
    """

    status: str
    parameters: ProjectionParametersModel
    active_model: ActiveModel | None = None
    nodes: list[ProjectionOutModelNode]


class FeatureModel(BaseModel):
    """
    Feature model
    """

    type: str
    name: str
    use_default_name: bool = True
    parameters: dict[str, str | float]


class LogisticL1Params(BaseModel):
    costLogL1: float


class LogisticL2Params(BaseModel):
    costLogL2: float


class KnnParams(BaseModel):
    n_neighbors: int


class RandomforestParams(BaseModel):
    n_estimators: int
    max_features: int | None


class Multi_naivebayesParams(BaseModel):
    alpha: float
    fit_prior: bool = True
    class_prior: str | None = None


class GenerationCreationModel(BaseModel):
    """
    GenAI model used in generation
    """

    slug: str
    api: str
    name: str
    endpoint: str | None = None
    credentials: str | None = None


class GenerationModel(GenerationCreationModel):
    """
    GenAI model used in generation
    """

    id: int


class BertopicParamsModel(BaseModel):
    """
    Parameters for BERTopic model
    https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#n_gram_range
    """

    language: str | None = None
    # min_topic_size: int | None = None # Removed because overridden by the hdbscan model - Axel
    top_n_words: int = 15
    n_gram_range: tuple[int, int] = (1, 2)
    # nr_topics: int | str = "auto" # Removed to propose topic reduction later in the pipeline - Axel
    outlier_reduction: bool = True
    hdbscan_min_cluster_size: int = 10
    umap_n_neighbors: int = 10
    umap_n_components: int = 2
    # umap_min_dist: float = 0.0 # Removed because 0.0 is the best value to use for clustering - Axel
    embedding_kind: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    filter_text_length: int = 2
    input_datasets: str = "train"


class ComputeBertopicModel(BertopicParamsModel):
    """
    Parameters for computing BERTopic model
    """

    name: str
    force_compute_embeddings: bool = False
    embedding_model: str
    language: str | None = None
    input_datasets: str = "train"
    umap_n_neighbors: int = 30
    hdbscan_min_cluster_size: int = 15
    outlier_reduction: bool = True
    filter_text_length: int = 50
    umap_n_components: int = 5
    top_n_words: int = 15
    n_gram_range: tuple[int, int] = (1, 2)
    embedding_kind: str = "sentence_transformers"
    scheme: str


class GenerationAvailableModel(BaseModel):
    """
    GenAI models available for generation
    """

    slug: str
    api: str
    name: str


class GenerationModelApi(BaseModel):
    """
    GenAI API available for generation
    """

    name: str
    models: list[GenerationAvailableModel]


class MLStatisticsModel(BaseModel):
    training_kind: str | None = None
    f1_label: dict[str, float] | None = None
    precision_label: dict[str, float] | None = None
    recall_label: dict[str, float] | None = None
    f1_weighted: float | None = None
    f1_micro: float | None = None
    f1_macro: float | None = None
    accuracy: float | dict[str, float] | None = None
    precision: float | dict[str, float] | None = None
    confusion_matrix: list[list[int]] | None = None
    false_predictions: dict[str, Any] | list[Any] | None = None
    table: dict[str, Any] | None = None


class GenerationRequest(BaseModel):
    """
    To start a generating prompt
    """

    model_id: int
    token: str | None = None
    prompt: str
    n_batch: int = 1
    scheme: str
    mode: str = "all"
    dataset: str = "train"
    prompt_name: str | None = None


class ProjectUpdateModel(BaseModel):
    project_name: str | None = None
    language: str | None = None
    cols_text: list[str] | None = None
    cols_context: list[str] | None = None
    add_n_train: int | None = None


# --------------------
# CLASS FOR COMPUTING
# --------------------


class ProcessComputing(BaseModel):
    user: str
    unique_id: str
    time: datetime.datetime
    kind: str


class UpdateComputing(ProcessComputing):
    update: ProjectUpdateModel


class LMComputing(ProcessComputing):
    model_name: str
    status: str
    scheme: Optional[str] = None
    dataset: Optional[str] = None
    get_progress: Callable[[], float | None] | None = None
    params: dict[str, Any] | None = None
    training_kind: str


class LMComputingOutModel(BaseModel):
    name: str
    status: str
    progress: float | None = None
    loss: dict[str, dict] | None = None
    epochs: float | None = None


class ProjectionComputing(ProcessComputing):
    kind: Literal["projection"]
    name: str
    method: str
    params: ProjectionParametersModel
    normalize_features: bool = False


class FeatureComputing(ProcessComputing):
    kind: Literal["feature"]
    name: str
    type: str
    parameters: dict


class GenerationComputing(ProcessComputing):
    kind: Literal["generation"]
    project: str
    number: int
    model_id: int
    dataset: str = "train"
    get_progress: Callable[[], float | None] | None = None
    prompt_name: str | None = None


class BertopicComputing(ProcessComputing):
    kind: Literal["bertopic"]
    name: str
    path_data: Path
    col_id: str | None
    col_text: str | None
    parameters: BertopicParamsModel
    force_compute_embeddings: bool
    get_progress: Callable[[], str | float | None] | None = None
    scheme: str  # This is a dummy necessary to save the model in the database, it will not be used afterwards — Axel


class QuickModelInModel(BaseModel):
    """
    Request Quickmodel
    TODO : model for parameters
    """

    name: str
    scheme: str
    model: str
    features: list
    params: dict[str, str | float | bool | list | None]
    standardize: bool | None = True
    dichotomize: str | None = None
    cv10: bool = False
    balance_classes: bool = False
    exclude_labels: list[str] = []
    test_size: float = 0.2


class QuickModelComputing(ProcessComputing):
    """
    Quickmodel object
    """

    status: Literal["training", "predicting"]
    name: str
    scheme: str
    features: list
    labels: list
    model_type: str
    model_params: dict
    dataset: str
    standardize: bool = False
    cv10: bool = False
    balance_classes: bool = False
    exclude_labels: list[str] = []
    test_size: float = 0.2
    retrain: bool = False


class QuickModelComputed(BaseModel):
    """
    Quickmodel object
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: str = "trained"
    name: str
    features: list
    scheme: str
    labels: list
    user: str
    model_type: str
    model_params: dict
    time: datetime.datetime
    standardize: bool = False
    cv10: bool = False
    balance_classes: bool = False
    exclude_labels: list[str] = []
    test_size: float = 0.2
    retrain: bool = False
    proba: DataFrame | None = None
    model: BaseEstimator
    statistics_train: MLStatisticsModel | None = None
    statistics_test: MLStatisticsModel | None = None
    statistics_cv10: MLStatisticsModel | None = None


class QuickModelOutModel(BaseModel):
    """
    Trained quickmodel
    """

    name: str
    features: list
    model: str
    params: (
        dict[str, str | float | bool | list | None]
        | dict[str, dict[str, str | float | bool | None]]
        | None
    )
    scheme: str
    username: str
    statistics_train: MLStatisticsModel | None = None
    statistics_test: MLStatisticsModel | None = None
    statistics_cv10: MLStatisticsModel | None = None
    balance_classes: bool = False
    exclude_labels: list[str] = []


class GenerationComputingOut(BaseModel):
    """
    Response for generation
    """

    model_id: int
    progress: float | None


class TableOutModel(BaseModel):
    """
    Response for table of elements
    """

    items: list
    total: int | float


class TableInModel(BaseModel):
    """
    Requesting a table of elements
    """

    list_ids: list
    list_labels: list
    scheme: str
    action: str


class TableBatchInModel(BaseModel):
    """
    Requesting a batch of elements
    """

    scheme: str
    min: int = 0
    max: int = 0
    contains: str | None = None
    dataset: str = "train"
    on_users: list[str] | None = None
    on_labels: list[str] | None = None
    recent: bool = False


class ProjectsServerModel(BaseModel):
    """
    Response for available projects
    """

    projects: list[str]
    auth: list


class ProjectSummaryModel(BaseModel):
    project_slug: str
    parameters: ProjectModel
    user_right: str
    created_by: str
    created_at: str
    size: float | None = None
    last_activity: str | None = None


class AvailableProjectsModel(BaseModel):
    """
    Response for available projects
    """

    projects: list[ProjectSummaryModel]
    storage_used: float | None = None  # in GB
    storage_limit: float | None = None  # in GB


## State definition of the project


class NextProjectStateModel(BaseModel):
    methods_min: list[str]
    methods: list[str]
    sample: list[str]


class SchemesProjectStateModel(BaseModel):
    available: dict[str, SchemeModel]


class FeaturesProjectStateModel(BaseModel):
    options: dict[str, dict[str, Any]]
    available: list[str]
    training: dict[str, dict[str, str | None]]


class ModelDescriptionModel(BaseModel):
    name: str
    kind: str
    scheme: str
    parameters: dict[str, Any]
    path: str
    time: str


class QuickModelsProjectStateModel(BaseModel):
    options: dict[str, Any]
    # available: dict[str, dict[str, QuickModelOutModel]]
    # training: dict[str, list[str]]
    available: dict[str, list[ModelDescriptionModel]]
    training: dict[str, list[str]]


class LanguageModelsProjectStateModel(BaseModel):
    options: list[dict[str, Any]]
    available: dict[str, dict[str, LMStatusModel]]
    training: dict[str, LMComputingOutModel]
    base_parameters: LMParametersModel


class ProjectionsProjectStateModel(BaseModel):
    options: dict[str, dict[str, Any]]
    available: dict[str, str | int]
    training: dict[str, str]


class BERTopicDescriptionModel(BaseModel):
    name: str
    time: str


class BertopicProjectStateModel(BaseModel):
    available: dict[str, BERTopicDescriptionModel]
    training: dict[str, dict[str, str | int | float | None]]
    models: list[str]


class GenerationsProjectStateModel(BaseModel):
    training: dict[str, GenerationComputingOut]


class ErrorsProjectStateModel(BaseModel):
    errors: list[list]


class UsersStateModel(BaseModel):
    users: list[str]
    last_schemes: dict[str, str]


class ProjectStateModel(BaseModel):
    """
    Response for server state
    """

    params: ProjectModel
    next: NextProjectStateModel
    schemes: SchemesProjectStateModel
    features: FeaturesProjectStateModel
    quickmodel: QuickModelsProjectStateModel
    languagemodels: LanguageModelsProjectStateModel
    projections: ProjectionsProjectStateModel
    generations: GenerationsProjectStateModel
    bertopic: BertopicProjectStateModel
    users: UsersStateModel
    errors: list[list]
    memory: float | None = None
    last_activity: str | None = None


class ProjectDescriptionModel(BaseModel):
    """
    Project description
    """

    users: list[str]
    train_set_n: int
    train_annotated_n: int
    train_annotated_distribution: dict[str, Any]
    test_set_n: int | None = None
    valid_set_n: int | None = None
    test_annotated_n: int | None = None
    valid_annotated_n: int | None = None
    test_annotated_distribution: dict[str, Any] | None = None
    valid_annotated_distribution: dict[str, Any] | None = None
    sm_10cv: Any | None = None


class ProjectAuthsModel(BaseModel):
    """
    Auth description for a project
    """

    auth: dict[str, str]


class WaitingModel(BaseModel):
    """
    Response for waiting
    """

    detail: str
    status: str = "waiting"


class DocumentationModel(BaseModel):
    """
    Documentation model
    """

    credits: list[str]
    page: str
    documentation: str
    contact: str


class ReconciliationModel(BaseModel):
    """
    list of elements to reconciliate
    """

    table: list[dict[str, str | dict[str, str | None] | None]]
    users: list[str]
    n_total: int = 0
    n_agreements: int = 0
    n_disagreements: int = 0
    agreement_percentage: float | None = None
    cohen_kappa: float | None = None


class ReconciliateElementInModel(BaseModel):
    """
    Reconciliate specific element
    """

    dataset: str
    scheme: str
    element_id: str
    label: str
    users: list[str]


class AuthActions(StrEnum):
    add = "add"
    delete = "delete"


class TableBatch(BaseModel):
    batch: DataFrame
    total: int
    min: int
    max: int
    filter: str | None

    class Config:
        arbitrary_types_allowed: bool = True  # Allow DataFrame type but switches off Pydantic here


class CodebookModel(BaseModel):
    content: str
    scheme: str
    time: str


class GenerationResult(BaseModel):
    user: str
    project_slug: str
    model_id: int
    element_id: str
    prompt: str
    answer: str


class GpuInformationModel(BaseModel):
    gpu_available: bool
    total_memory: float
    available_memory: float


class MessagesInModel(BaseModel):
    kind: str
    content: str
    for_project: str | None = None
    for_user: str | None = None


class MessagesOutModel(BaseModel):
    id: int
    kind: str
    created_by: str
    time: str
    content: str


class ServerStateModel(BaseModel):
    version: str
    cpu_only: bool = False
    queue: dict[str, dict[str, str | None]]
    active_projects: dict[str, list]
    gpu: GpuInformationModel
    cpu: dict
    memory: dict
    disk: dict
    mail_available: bool
    messages: list[MessagesOutModel]


class StaticFileModel(BaseModel):
    name: str
    path: str


class ProjectStaticFiles(BaseModel):
    dataset: StaticFileModel
    model: StaticFileModel | None = None


class FeatureDescriptionModel(BaseModel):
    name: str
    parameters: dict[str, Any]
    user: str
    time: str
    kind: str
    cols: list[str]


class FeatureDescriptionModelOut(BaseModel):
    name: str
    parameters: dict[str, Any]
    user: str
    time: str
    kind: str


class TrainMLResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    proba: DataFrame
    statistics: MLStatisticsModel
    statistics_cv10: MLStatisticsModel | None = None


class ReturnTaskPredictModel(BaseModel):
    path: str
    metrics: dict[str, MLStatisticsModel] | None = None


class EventsModel(BaseModel):
    events: dict[str, dict[str, str | None]]


class ModelScoresModel(BaseModel):
    internalvalid_scores: dict | None = None
    train_scores: dict | None = None
    valid_scores: dict | None = None
    test_scores: dict | None = None
    outofsample_scores: dict | None = None


class ModelInformationsModel(BaseModel):
    params: dict | None = None
    loss: dict | None = None
    scores: ModelScoresModel


class UserStatistics(BaseModel):
    username: str
    projects: dict[str, str]
    # last_connexion
    # last_actions


class PromptInputModel(BaseModel):
    text: str
    name: str | None = None


class PromptModel(BaseModel):
    id: int
    text: str
    parameters: dict[str, Any]


class TextDatasetModel(BaseModel):
    id: str
    text: str
    filename: str
    path: Path | None = None


class GeneratedElementsIn(BaseModel):
    n_elements: int
    filters: list[str] = []


class ExportGenerationsParams(BaseModel):
    filters: list[str] = []


class ProjectCreatingModel(BaseModel):
    project_slug: str
    username: str
    unique_id: str
    time: datetime.datetime
    kind: str
    status: str


class TopicsOutModel(BaseModel):
    Topic: int
    Name: str
    Count: int
    Representation: str
    Representative_Docs: str


class BertopicOutModelParameters(BaseModel):
    bertopic_params: ComputeBertopicModel
    col_text: str
    col_id: str | None
    name: str
    timestamp: str  # Not sure about this one, example: 20251027_104836 # Axel
    path_data: str
    path_embeddings: str
    path_projection: str


class BertopicTopicsOutModel(BaseModel):
    topics: list[TopicsOutModel]
    parameters: BertopicOutModelParameters


class DatasetModel(BaseModel):
    """
    Datasets authorized for a user
    """

    project_slug: str
    columns: list[str]
    n_rows: int


class AuthUserModel(BaseModel):
    """
    Information on auth
    """

    project_slug: str
    username: str
    status: str | None = None


class MonitoringQuickModelsModel(BaseModel):
    """
    Monitoring quickmodels
    """

    n: int
    mean: float
    std: float


class MonitoringLanguageModelsModel(BaseModel):
    """
    Monitoring language models
    """

    n: int
    mean: float
    std: float


class MonitoringMetricsModel(BaseModel):
    """
    Monitoring metrics
    """

    quickmodels: MonitoringQuickModelsModel
    languagemodels: MonitoringLanguageModelsModel


class BertopicProjectionNode(BaseModel):
    """
    Node metadata
    """

    x: float
    y: float
    cluster_id: int
    label: str
    node_id: str


class BertopicProjectionData(BaseModel):
    """
    The returned data when fetching the projection for a topic analysis
    """

    nodes: list[BertopicProjectionNode]
    cluster_id_label_mapper: dict
