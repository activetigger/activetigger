from datetime import datetime, timezone

import pandas as pd

from activetigger.datamodels import (
    EventsModel,
    MonitoringLanguageModelsModel,
    MonitoringMetricsModel,
    MonitoringQuickModelsModel,
)
from activetigger.db.manager import DatabaseManager


class TaskTimer:
    """This object centralises the timing component in order to save them as part
    of the "additional_event" in the Monitoring.close_process function"""

    body = {"start": "FAILED", "end": "FAILED", "duration": "FAILED", "order": None}

    def __init__(
        self, compulsory_steps: list[str], optional_steps: list[str] | None = None
    ) -> None:
        self.__additional_events = {step: self.body for step in compulsory_steps}
        self.__starts: dict[str, datetime] = {}
        self.__stops: list[str] = []
        self.__optional_steps: list[str] = optional_steps if optional_steps is not None else []

    def start(self, step: str) -> None:
        """
        Starts the corresponding timer. Make sure that the step exists, if
        optional, initiate the step body.
        """

        if step in self.__optional_steps:
            self.__additional_events[step] = self.body
        if step not in self.__additional_events:
            raise Exception(
                (
                    f"TaskTimer.start(step): {step} is not one of the compulsory "
                    f"steps ({self.__additional_events.keys()})"
                )
            )
        if step in self.__starts:
            raise Exception((f"TaskTimer.start(step): {step} timer has already been started."))
        self.__starts[step] = datetime.now(timezone.utc)

    def stop(self, step: str) -> None:
        """
        Stops the timer
        """

        if step not in self.__starts:
            raise Exception(
                (
                    f"TaskTimer.stop(step): the step {step} timer was not started "
                    f"or previously stopped."
                )
            )
        if step in self.__stops:
            raise Exception(
                (f"TaskTimer.stop(step): the step {step} timer has already been stopped.")
            )

        end = datetime.now(timezone.utc)
        self.__stops += [str(step)]
        self.__additional_events[step] = {
            "start": self.__starts[step].isoformat(),
            "end": end.isoformat(),
            "duration": str((end - self.__starts[step]).total_seconds()),
            "order": str(len(self.__stops)),
        }

    def get_events(self) -> dict[str, dict[str, str | None]]:
        return self.__additional_events


class Monitoring:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    db_manager: DatabaseManager
    project_slug: str | None

    def __init__(self, db_manager: DatabaseManager, project_slug: str | None = None) -> None:
        self.db_manager = db_manager
        self.project_slug = project_slug

    def register_process(
        self, process_name: str, kind: str, parameters: dict, user_name: str
    ) -> None:
        """
        Start a new monitored process
        """
        events = {"global": {"start": datetime.now(timezone.utc).isoformat()}}
        self.db_manager.monitoring_service.add_process(
            process_name=process_name,
            kind=kind,
            parameters=parameters,
            events=events,
            project_slug=self.project_slug,
            user_name=user_name,
        )

    def close_process(self, process_name: str, list_events: EventsModel) -> None:
        """
        Close a monitored process
        """
        start_entry = self.db_manager.monitoring_service.get_element_by_process(process_name)
        if start_entry is None:
            raise ValueError(f"Process {process_name} not found")

        # Save the duration of the global process
        events = start_entry.events
        end = datetime.now(timezone.utc)
        events["global"]["end"] = end.isoformat()
        # Ensure start_entry.time is timezone-aware for subtraction
        # (PostgreSQL returns tz-aware datetimes, SQLite returns naive ones)
        start_time = start_entry.time
        if start_time.tzinfo != end.tzinfo:
            start_time = start_time.astimezone(end.tzinfo)
        duration = (end - start_time).total_seconds()
        events["global"]["duration"] = duration
        events["global"]["order"] = -1

        # Add additional events to the events object
        additional_events = (
            list_events.events if list_events is not None and len(list_events.events) > 0 else None
        )
        if isinstance(additional_events, dict):
            # If additional events are passed on
            if ("start" in additional_events.keys()) or ("end" in additional_events.keys()):
                # We do not want to overwrite the "global"; keys, so we remove them prior to merging
                additional_events = {
                    key: value for key, value in additional_events.items() if key != "global"
                }
            # Merge the events with the additional events
            events.update(additional_events)

        self.db_manager.monitoring_service.update_process(
            process_name=process_name,
            events=events,
            duration=duration,
        )
        print(f"Process {process_name} closed in {duration} seconds")

    def get_completed_processes(
        self, kind: str, username: str | None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Get completed processes of a given kind and user
        """
        processes = self.db_manager.monitoring_service.get_completed_processes(
            kind=kind, username=username, limit=limit
        )
        data = [
            {
                "process_name": process.process_name,
                "kind": process.kind,
                "time": process.time,
                "parameters": process.parameters,
                "events": process.events,
                "project_slug": process.project_slug,
                "user_name": process.user_name,
                "duration": process.duration,
                "username": process.user_name,
            }
            for process in processes
        ]
        df = pd.DataFrame(data)
        return df

    def get_failed_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_running_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_data(self, kind: str, state: str = "completed", limit: int = 100) -> pd.DataFrame:
        """
        Get monitoring data
        """
        if state == "completed":
            df_processes = self.get_completed_processes(kind=kind, username=None, limit=limit)
        else:
            raise NotImplementedError("Not implemented yet")
        return df_processes

    def get_metrics(self) -> MonitoringMetricsModel:
        """
        Get monitoring metrics
        """
        df_quickmodels = self.get_completed_processes(
            kind="train_quickmodel", username=None, limit=100
        )
        m_quickmodels = MonitoringQuickModelsModel(
            n=len(df_quickmodels),
            mean=0 if len(df_quickmodels) == 0 else df_quickmodels["duration"].mean(),
            std=0 if len(df_quickmodels) < 2 else df_quickmodels["duration"].std(),
        )
        df_languagemodels = self.get_completed_processes(
            kind="train_languagemodel", username=None, limit=100
        )
        m_languagemodels = MonitoringLanguageModelsModel(
            n=len(df_languagemodels),
            mean=0 if len(df_languagemodels) == 0 else df_languagemodels["duration"].mean(),
            std=0 if len(df_languagemodels) < 2 else df_languagemodels["duration"].std(),
        )
        return MonitoringMetricsModel(quickmodels=m_quickmodels, languagemodels=m_languagemodels)
