# Active Tigger

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/activetigger/activetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![React](https://img.shields.io/badge/React-18.3-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.4-3178C6)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

Hi❗

ActiveTigger[^1] is a collaborative text annotation web tool dedicated to computational social sciences. It is designed to assist exploration and model (BERT) fine-tuning to classify text datasets relying on active learning.

Designed primarily by researchers in social sciences, its use can extend to all users that need to annotate textual data.

> [!IMPORTANT]
> The app is currently 🚧 **in beta test** 🚧. The stable version is planned for May 2026.


## Run the app with Docker (recommended)
 
:warning: If you don't already have it installed, install docker/docker compose first :whale:. You need the docker daemon running to proceed.

Clone the current repository.

```bash
git clone https://github.com/activetigger/activetigger.git
```

Favor the production branch.

```bash
cd activetigger
git checkout production
```

Then, you can run the app with docker compose. You need to be in the `docker` directory.

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.dev.yml -p activetigger up
```

The configuration file is `./docker/.env` to set the environment variables. If you want to change the default `root` password you need to modify it.

Docker will start:

- Postgresql
- Python API (and install all requirements)
- frontend
- a reverse proxy with Nginx

_By default the docker stack is in mode DEV._ This means that both API and FRONTEND are started using the local code in watch mode. You can update the code and the service will be restarted. You can therefore use docker for development.

> [!IMPORTANT]
> If you want to use a GPU inside docker locally you need to first follow the `NVIDIA GPU` section from the [deploy documentation](./DEPLOY.md#nvidia-gpu).

## Install the app without Docker

The app is built on a client/API/Database architecture :

- the server runs an API with FastAPI
- the client is in [React](https://reactjs.org/)
- The database can be either SQLite or Postgresql (if you use Postgresql, you need to install it first). For local non-docker installation, it is recommended to use SQLite.

First, clone the repository

```bash
git clone https://github.com/activetigger/activetigger.git
```

### Install the Python API

Add a specific `config.yaml` file in the api directory if you want to specify the path of the static files and database (you can modify and rename the `config.yaml.sample` or use the default config):

- `path` : path to store files (for instance `./data`)
- `path_models` : path to store the models (for instance `./data/models`)

#### Option 1: Using uv (recommended)

We recommend using [uv](https://docs.astral.sh/uv/), a fast Python package and project manager written in Rust. It replaces `pip`, `venv`, and `conda` in a single tool and automatically handles Python version management, virtual environments, and dependency resolution — making setup faster and more reliable.

Install uv on macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See the [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for more options (Homebrew, pip, etc.).

Then, from the `api/` directory, `uv` will automatically create a virtual environment and install all dependencies (including the correct Python version):

```bash
cd activetigger/api
uv sync
```

Launch the server (on 0.0.0.0 port 5000 by default, you can configure exposed port if needed with -p PORTNUM):

```bash
uv run python -m activetigger
```

#### Option 2: Using Anaconda/pip

Alternatively, create a virtual environment with Python 3.11 using Anaconda/Miniconda ([documentation here for Ubuntu](https://doc.ubuntu-fr.org/miniconda)):

```bash
conda create -n activetigger python=3.11
conda activate activetigger
```

Install requirements:

```bash
pip install -r activetigger/api/requirements.txt
```

Launch the server:

```bash
cd activetigger/api
python -m activetigger
```

#### Check the API

Check that the API is running by going to `http://localhost:5000` in your browser.

#### Optional: Install GPU Support for UMAP

To enable GPU support, [install RAPIDS cuML](https://docs.rapids.ai/install/). For instance, for CUDA 12

```bash
pip install --extra-index-url https://pypi.nvidia.com cuml-cu12
```

#### Troubleshooting

On Ubuntu, you need to install some drivers for the Postgresql database and the python package psycopg2.

```bash
sudo apt-get install libpq-dev
```

### Install the React frontend

The frontend is written in React/Typescript. To run the dev version and to build the app, you need first to install node.js and npm (version > 20).

For linux :

```bash
sudo apt-get install nodejs npm
```

For mac, you can install brew https://brew.sh/ and

```bash
brew install node
```

Then you can install the npm packages

```bash
cd frontend
npm i
```

You can then run the dev version

```bash
npm run dev
```

If you run the backend on a different port, mind the fact that you need to change the address in the `frontend/.env` file to set the correct port.

To compile

```bash
npm run compile
```

To build

```bash
npm run build
```

You can deploy the app with Github Pages for tests



## Documentation

The documentation is [here](http://activetigger.com/activetigger/)

## Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md) before opening a pull request. The short version: **open an issue first** to discuss what you want to change, then submit a focused PR that addresses that single issue.

## Funding

The development of Active Tigger is supported by : [DRARI Île-de-France](https://www.enseignementsup-recherche.gouv.fr/fr/drari-ile-de-france) [ECODEC](https://labex-ecodec.ensae.fr/) [Progedo](https://www.progedo.fr/)

## How to cite

Boelaert J., Ollion É., Schultz É. (2026). ActiveTigger (Version 0.9.9) [Computer software]. https://github.com/activetigger/activetigger


[^1]: The current version is a refactor of [R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger). Active Tigger name is a pun that draws on the similarity between the words 'Tagger' and 'Tigger.'
