"""Phase 1: dataset acquisition.

Downloads the Kaggle/ULB Credit Card Fraud Detection dataset (284,807 transactions,
492 fraudulent, 30 features) via kagglehub and places it at config.RAW_DATA_FILE.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (create one at
https://www.kaggle.com/settings -> API -> Create New Token) before running.
"""

import shutil
from pathlib import Path

from src import config


def download_dataset(force: bool = False) -> Path:
    """Download creditcard.csv into data/raw/ if not already present.

    Returns the path to the local CSV file.
    """
    config.ensure_directories()

    if config.RAW_DATA_FILE.exists() and not force:
        return config.RAW_DATA_FILE

    import kagglehub

    download_dir = Path(kagglehub.dataset_download(config.KAGGLE_DATASET_ID))
    source_csv = next(download_dir.glob("*.csv"))
    shutil.copy(source_csv, config.RAW_DATA_FILE)
    return config.RAW_DATA_FILE


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset available at: {path}")
