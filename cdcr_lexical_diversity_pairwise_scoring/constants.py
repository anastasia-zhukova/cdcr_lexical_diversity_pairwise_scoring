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
MLFLOW_LOG_ARTIFACT_DIR = "logs"
MLFLOW_RUN_LOG_FILENAME = "run.log"
DEFAULT_RATIO = 20
DEFAULT_TRAIN = 40000000
DEFAULT_DEV = 8000
SENT_TRANSFOMER = "intfloat/multilingual-e5-large-instruct"
EMBEDDING =  "neuml/fasttext"
ENCODE_BATCH = 8
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
CONFIG_NAME = "test_random"