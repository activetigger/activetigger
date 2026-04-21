import json

import requests

from activetigger.generation.client import GenerationModelClient


class OpenAPI(GenerationModelClient):
    endpoint: str

    def __init__(self, endpoint: str, credentials: str | None):
        self.endpoint = endpoint
        self.credentials = credentials

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to OpenAPI
        """
        headers = {
            "Authorization": f"Bearer {self.credentials}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.endpoint,
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
