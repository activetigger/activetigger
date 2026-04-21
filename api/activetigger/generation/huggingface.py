from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.errors import HTTPError

from activetigger.generation.client import GenerationModelClient


class HuggingFace(GenerationModelClient):
    endpoint: str
    credentials: str | None
    client: InferenceClient

    def __init__(self, endpoint: str | None, credentials: str | None):
        self.endpoint = endpoint or ""
        self.credentials = credentials
        self.client = InferenceClient(token=credentials, timeout=30)

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to a HuggingFace model
        """
        try:
            response = self.client.text_generation(
                f"Your are a careful assistant who annotates texts for a research project in a JSON format. You follow precisely the guidelines, which can be in different languages. {prompt}",
                model=self.endpoint if self.endpoint != "" else model,
                max_new_tokens=50,
            )
        except InferenceTimeoutError as te:
            raise Exception from te
        except HTTPError as e:
            msg = e.response.content.decode("utf-8")  # ty: ignore[unresolved-attribute]
            raise Exception(f"HTTP error from endpoint: {msg}") from e
        except Exception as e:
            raise Exception from e

        return response
