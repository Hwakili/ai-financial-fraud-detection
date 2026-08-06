"""Central paths, constants, and random seed shared across all pipeline phases.

Kept in one place so every phase (EDA, preprocessing, baselines, RXT, evaluation,
XAI, efficiency benchmarking) agrees on the same data locations and the same
random seed - required for the results to be reproducible and comparable.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"

RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"

KAGGLE_DATASET_ID = "mlg-ulb/creditcardfraud"

RANDOM_SEED = 42

TARGET_COLUMN = "Class"
SCALED_COLUMNS = ["Time", "Amount"]

TEST_SIZE = 0.2


def ensure_directories() -> None:
    """Create every data/results subdirectory if it doesn't already exist."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        RESULTS_FIGURES_DIR,
        RESULTS_METRICS_DIR,
        RESULTS_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
