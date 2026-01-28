from pathlib import Path


# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent
DEFAULT_RATIO = 20
DEFAULT_TRAIN = 40000
DEFAULT_DEV = 8000
SENT_TRANSFOMER = "intfloat/multilingual-e5-large-instruct"
ENCODE_BATCH = 8
DELTA = 0.4