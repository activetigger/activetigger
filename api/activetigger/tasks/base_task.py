import uuid
from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self):
        """Initialize the task with given parameters."""
        self.unique_id = str(uuid.uuid4())
        self.event = None

    @abstractmethod
    def __call__(self):
        """The function that realize the task processing."""
        pass
