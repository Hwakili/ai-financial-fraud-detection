"""Sanity checks for src/config.py — paths resolve correctly and are creatable."""

from src import config


def test_project_root_contains_src():
    assert (config.PROJECT_ROOT / "src").is_dir()


def test_ensure_directories_creates_expected_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path / "results" / "figures")
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path / "results" / "metrics")
    monkeypatch.setattr(config, "RESULTS_MODELS_DIR", tmp_path / "results" / "models")

    config.ensure_directories()

    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert (tmp_path / "results" / "figures").is_dir()
    assert (tmp_path / "results" / "metrics").is_dir()
    assert (tmp_path / "results" / "models").is_dir()


def test_random_seed_is_fixed_for_reproducibility():
    assert isinstance(config.RANDOM_SEED, int)
