import json

import requests

from activetigger.generation.client import GenerationModelClient


class OpenAPI(GenerationModelClient):
    endpoint: str

    def __init__(self, endpoint: str, credentials: str | None):
        self.endpoint = endpoint
        self.credentials = credentials

    @staticmethod
    def _chat_url(endpoint: str) -> str:
        """Normalize a user-provided endpoint to a chat-completions URL."""
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        return url

    @staticmethod
    def _models_url(endpoint: str) -> str:
        """Derive the /models URL from a user-provided endpoint."""
        url = endpoint.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        return url + "/models"

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to an OpenAI-compatible chat-completions endpoint.
        """
        headers = {
            "Authorization": f"Bearer {self.credentials}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self._chat_url(self.endpoint),
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
            ),
            headers=headers,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(
                f"HTTP call responded with code {response.status_code}" + str(response.content)
            )

    @staticmethod
    def list_models(endpoint: str, credentials: str | None) -> list[dict[str, str]]:
        """
        Query an OpenAI-compatible server to list available models via /v1/models.
        """
        url = OpenAPI._models_url(endpoint)
        headers = {}
        if credentials:
            headers["Authorization"] = f"Bearer {credentials}"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch models from {url} (HTTP {response.status_code})")
        data = response.json()
        models = data.get("data", [])
        return [{"slug": m["id"], "name": m["id"]} for m in models]
