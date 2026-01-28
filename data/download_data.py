import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("ryanmouton/ohiot1dm")

print("Downloaded to:", path)

# Move to project data directory
target_dir = Path("diabetes_rl/data/raw/ohio_t1dm")
target_dir.mkdir(parents=True, exist_ok=True)

shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("Dataset copied to:", target_dir)
