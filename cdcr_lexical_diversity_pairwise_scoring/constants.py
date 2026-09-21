import os
from pathlib import Path

from dotenv import load_dotenv


# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent

# Environment (.env is git-ignored, see .env.example)
load_dotenv(PROJECT_ROOT / ".env")

# MLflow tracking: an empty/unset tracking URI disables tracking entirely
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "cdcr_lexical_diversity_pairwise_scoring")
MLFLOW_RUN_ID_KEY = "mlflow_run_id"
# Log files written next to the console output: both logging systems of the project land in them
LOG_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOGURU_TO_STDLIB_FORMAT = "{name}:{function}:{line} - {message}"
# Artifact directories of a run: one per pipeline step plus the captured log
MLFLOW_LOG_ARTIFACT_DIR = "logs"
MLFLOW_RUN_LOG_EXTENSION = ".log"
MLFLOW_TRAINING_ARTIFACT_DIR = "training"
MLFLOW_INFERENCE_ARTIFACT_DIR = "inference"
MLFLOW_SCORING_ARTIFACT_DIR = "scoring"
DEFAULT_RATIO = 20
DEFAULT_TRAIN = 40000000
DEFAULT_DEV = 8000
SENT_TRANSFOMER = "intfloat/multilingual-e5-large-instruct"
EMBEDDING =  "neuml/fasttext"
ENCODE_BATCH = 8
# new vectors of the contrastive pair sampling are written to their .h5 cache in batches of this size
VECTOR_CACHE_SAVE_EVERY = 5000
ALLOWED_TOPICS = {
    # "WECEng": ["Meetings", "Civilian Attack", "Airliner Accident", "Earthquake", "News Event", "Terrorist Attack", "Wildfire", "Flood", "Weapons Test", "Eruption", "Oilspill", "Rail Accident"],
    # # "HyperCoref": ["gma", "international", "politics", "business", "blotter", "health", "us", "abc_univision", "thetorldnewser", 'thelaw']
    # "HyperCoref": ["international", "politics", "us"]
}
MIN_STD = 0.03
DENOM_DELTA = 4
MAX_ALLOWED_BATCH_SIZE = 200
CLUSTERING_THRESHOLD = 0.5
EXCLUDE_SINGLETONS = True
SCORE_ALL_EXPERIMENTS = True
# Experiment config used when a script is started without --config-name
DEFAULT_CONFIG_NAME = "test_random"