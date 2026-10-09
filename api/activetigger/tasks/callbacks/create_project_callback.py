from os import unlink

import pandas as pd
from typing_extensions import override

from activetigger.datamodels import ProjectModel

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.create_project_task import (
    CreateProjectTask,
    CreateProjectTaskResult,
)


class CreateProjectCallback(TaskCallback):
    # task_name is used to identify which callback to execute
    task_name = CreateProjectTask.name

    # the slug is nested in the dumped project model
    @classmethod
    @override
    def get_project_slug(cls, task_result):
        return task_result["project"]["project_slug"]

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    @override
    def on_complete_task_specific(
        cls, task_id: str, task_result: CreateProjectTaskResult, project_manager, process
    ):
        # create project model from the dump version
        project = ProjectModel(**task_result["project"])
        # load DataFrames from filesystem
        trainset_import = (
            pd.read_parquet(task_result["import_trainset_path"])
            if task_result["import_trainset_path"]
            else None
        )
        testset_import = (
            pd.read_parquet(task_result["import_testset_path"])
            if task_result["import_testset_path"]
            else None
        )
        validset_import = (
            pd.read_parquet(task_result["import_validset_path"])
            if task_result["import_validset_path"]
            else None
        )

        project_manager.finish_project_creation(
            task_result["username"], project, trainset_import, testset_import, validset_import
        )
        # clean filesystem
        if task_result["import_trainset_path"]:
            unlink(task_result["import_trainset_path"])
        if task_result["import_testset_path"]:
            unlink(task_result["import_testset_path"])
        if task_result["import_validset_path"]:
            unlink(task_result["import_validset_path"])
