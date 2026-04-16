import os
import shutil
from pathlib import Path

data_folder = "./test-data"

os.environ["DATA_PATH"] = data_folder
os.environ["ROOT_PASSWORD"] = "l3tm31n!"

path = Path(data_folder)
shutil.rmtree(path, ignore_errors=True)
os.mkdir(path)
os.mkdir(path / "projects")
os.mkdir(path / "projects" / "static")
os.mkdir(path / "models")
