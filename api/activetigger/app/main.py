import importlib.resources
import logging
import time
from contextlib import asynccontextmanager
from importlib.abc import Traversable
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from activetigger import __version__
from activetigger.app.dependencies import ProjectAction, ServerAction, test_rights, verified_user
from activetigger.app.routers import (
    annotations,
    bertopic,
    export,
    features,
    files,
    generation,
    messages,
    models,
    monitoring,
    projects,
    schemes,
    users,
)
from activetigger.config import config
from activetigger.datamodels import (
    ServerStateModel,
    TableOutModel,
    TokenModel,
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Frame the execution of the api
    """
    orchestrator = get_orchestrator()
    app.state.orchestrator = orchestrator
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")


# starting the app
app = FastAPI(lifespan=lifespan, root_path="/api")

# setup file logger for fastapi events
log_dir = Path(config.data_path) / "projects" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir.joinpath("fastapi_events.log")

fastapi_logger = logging.getLogger("activetigger.fastapi")
fastapi_logger.setLevel(logging.INFO)
fastapi_logger.propagate = False

if not fastapi_logger.handlers:
    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="H",
        interval=1,
        backupCount=24,
        encoding="utf-8",
        utc=False,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z",
        )
    )
    fastapi_logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    warning_delay = 1000
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        fastapi_logger.info(
            '%s "%s %s" status=%s duration_ms=%s %s',
            client,
            method,
            path,
            response.status_code,
            duration_ms,
            "DELAY" if duration_ms > warning_delay else "",
        )
        return response
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        fastapi_logger.exception(
            '%s "%s %s" status=500 duration_ms=%s PB',
            client,
            method,
            path,
            duration_ms,
        )
        raise


# add static folder
app.mount(
    "/static", StaticFiles(directory=Path(config.data_path) / "projects" / "static"), name="static"
)

# add routers
app.include_router(users.router)
app.include_router(projects.router)
app.include_router(annotations.router)
app.include_router(schemes.router)
app.include_router(features.router)
app.include_router(export.router)
app.include_router(models.router)
app.include_router(generation.router)
app.include_router(files.router)
app.include_router(bertopic.router)
app.include_router(messages.router)
app.include_router(monitoring.router)


# allow multiple servers (avoir CORS error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------
# Generic routes
# --------------


@app.get("/", response_class=HTMLResponse)
def welcome() -> str:
    """
    Welcome page for the API
    """
    data_path: Traversable = importlib.resources.files("activetigger")
    with open(str(data_path.joinpath("html", "welcome.html")), "r") as f:
        r = f.read()
    return r


@app.get("/version")
def get_version() -> str:
    """
    Get the version of the server
    """
    return __version__


@app.post("/server/restart", dependencies=[Depends(verified_user)])
def restart_queue(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Restart the queue & the memory
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    try:
        get_orchestrator().reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server")
def get_queue() -> ServerStateModel:
    """
    Get the state of the server
    """
    return get_orchestrator().server_state


@app.post("/token")
def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenModel:
    """
    Authentificate user from username/password and return token
    """
    try:
        orchestrator = get_orchestrator()
        user = orchestrator.users.authenticate_user(form_data.username, form_data.password)
        access_token = orchestrator.create_access_token(
            data={"sub": user.username}, expires_min=120
        )
        return TokenModel(access_token=access_token, token_type="bearer", status=user.status)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@app.get("/logs", dependencies=[Depends(verified_user)])
def get_logs(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str = "all",
    limit: int = 100,
) -> TableOutModel:
    """
    Get all logs for a username/project
    """
    if project_slug == "all":
        test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    else:
        test_rights(ProjectAction.GET, current_user.username, project_slug)
    df = get_orchestrator().get_logs(project_slug, limit)
    return TableOutModel(
        items=df.to_dict(orient="records"),
        total=limit,
    )


@app.post("/stop", dependencies=[Depends(verified_user)])
def stop_process(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    unique_id: str | None = None,
    project_slug: str | None = None,
    kind: str | None = None,
) -> None:
    """
    Stop processes either by unique_id or by kind for a user
    - unique_id: stop a specific process (only for administrator)
    - kind: stop all processes of a given kind for the user
    """
    if unique_id is None and kind is None:
        raise HTTPException(status_code=400, detail="You must provide a unique_id or a kind")
    try:
        orchestrator = get_orchestrator()
        if unique_id is not None:
            test_rights(ServerAction.MANAGE_SERVER, current_user.username)
            orchestrator.stop_process(unique_id, current_user.username)
        if project_slug is not None:
            # rights already checked
            orchestrator.stop_user_processes(current_user.username, project_slug, kind)
        orchestrator.log_action(
            current_user.username,
            f"STOP PROCESS: {kind if kind is not None else unique_id}",
            "general",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
