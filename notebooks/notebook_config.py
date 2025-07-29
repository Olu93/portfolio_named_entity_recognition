import pathlib
import sys

WORK_DIR = (pathlib.Path(__file__).parent.parent).absolute()
sys.path.append(str(WORK_DIR / "src"))



NOTEBOOKS_DIR = WORK_DIR / "notebooks"
FILES_DIR = WORK_DIR / "files"
DATASETS_DIR =FILES_DIR / "datasets"
EXPERIMENTAL_RESULTS_DIR = FILES_DIR / "experimental_results"
MISC_DIR = FILES_DIR / "misc"
MODELS_DIR = WORK_DIR / "models"

from constants.model_config import MODEL_CONFIGS

# Set python path to the root of the project
# sys.path.append(str(WORK_DIR))

# Set src path to the root of the project so notebooks can import from src modules





