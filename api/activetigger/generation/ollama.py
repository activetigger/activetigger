import requests

from activetigger.generation.client import GenerationModelClient


class Ollama(GenerationModelClient):
    endpoint: str

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @staticmethod
    def _generate_url(endpoint: str) -> str:
        """Ensure the endpoint points to /api/generate"""
        url = endpoint.rstrip("/")
        if not url.endswith("/api/generate"):
            if url.endswith("/api"):
                url += "/generate"
            else:
                url += "/api/generate"
        return url

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to ollama
        """
        m = model if model is not None else "llama3.3:70b"
        url = self._generate_url(self.endpoint)
        data = {"model": m, "prompt": prompt, "stream": False}
        response = requests.post(url, json=data, verify=False, timeout=60)
        if response.status_code != 200:
            raise Exception(f"HTTP call responded with code {response.status_code}")
        result = response.json()
        if "error" in result:
            raise Exception(f"Ollama error: {result['error']}")
        return result["response"]

    @staticmethod
    def list_models(endpoint: str) -> list[dict[str, str]]:
        """
        Query an Ollama server to list available models via /api/tags
        """
        # Derive the base URL from the endpoint (which may point to /api/generate)
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/api/generate"):
            base_url = base_url[: -len("/api/generate")]
        elif base_url.endswith("/api"):
            base_url = base_url[: -len("/api")]

        tags_url = base_url + "/api/tags"
        response = requests.get(tags_url, verify=False, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch models from {tags_url} (HTTP {response.status_code})")
        data = response.json()
        models = data.get("models", [])
        return [{"slug": m["name"], "name": m["name"]} for m in models]
