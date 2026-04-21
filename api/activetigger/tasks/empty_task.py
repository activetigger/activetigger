import time

from activetigger.tasks.base_task import BaseTask


class EmptyTask(BaseTask):
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        super().__init__()

    def __call__(self) -> bool:
        """
        wait
        """
        for i in range(1, self.timeout):
            print(f"Time process : {i} ")
            time.sleep(1)
        return True
