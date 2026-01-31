from pathlib import Path


# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent
DEFAULT_RATIO = 20
DEFAULT_TRAIN = 40000
DEFAULT_DEV = 8000
SENT_TRANSFOMER = "intfloat/multilingual-e5-large"
ENCODE_BATCH = 8
DELTA = 0.4
ALLOWED_TOPICS = {
    "WECEng": ["Meetings", "Civilian Attack", "Airliner Accident", "Earthquake", "News Event", "Terrorist Attack", "Wildfire", "Flood", "Weapons Test", "Eruption", "Oilspill", "Rail Accident"],
    "HyperCoref": ["gma", "international", "politics", "business", "blotter", "health", "us", "abc_univision", "thetorldnewser", 'thelaw']
    # "HyperCoref": ["international", "politics", "us", "blotter"]
}
DENOM_DELTA = 4