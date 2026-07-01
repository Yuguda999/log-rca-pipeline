from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

LOGS_CSV = DATA_DIR / "logs.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

TEXT_COL = "log_message"
LABEL_COL = "root_cause_label"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Predictions below this probability are flagged for human review.
CONFIDENCE_THRESHOLD = 0.45

# Local LLM (Ollama) settings for the optional summary path.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(exist_ok=True)
