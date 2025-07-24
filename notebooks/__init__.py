import pathlib
import sys

WORK_DIR = (pathlib.Path(__file__).parent.parent).absolute()
NOTEBOOKS_DIR = WORK_DIR / "notebooks"
FILES_DIR = WORK_DIR / "files"

# Set python path to the root of the project
# sys.path.append(str(WORK_DIR))

# Set src path to the root of the project
sys.path.append(str(WORK_DIR / "src"))





