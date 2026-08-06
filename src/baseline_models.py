"""Phase 3: baseline classifiers — Logistic Regression, Random Forest, XGBoost.

These establish the performance floor that the Phase 4 RXT model must clearly
beat to justify its added architectural complexity (per the ToR's evaluation
plan: "Advanced deep learning models will be benchmarked against traditional
classifiers to quantify performance gains and justify architectural complexity").
"""

import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.evaluate import evaluate_model, save_predictions


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """scale_pos_weight = negative:positive ratio - XGBoost's equivalent of
    class_weight='balanced', since it has no native 'balanced' option."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all_baselines(X_train, y_train) -> dict:
    return {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }


def run_baseline_pipeline(X_train, y_train, X_test, y_test, persist_models: bool = True) -> tuple:
    """Train every baseline on the given (already imbalance-handled) training
    set, evaluate each on the untouched test set via the shared evaluate.py
    pipeline, and return (models_dict, metrics_dict).

    Models are also persisted to results/models/ (joblib) so Phase 6 (SHAP)
    and Phase 7 (efficiency) can reload them without retraining.
    """
    models = train_all_baselines(X_train, y_train)
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate_model(name, y_test, y_pred, y_proba)
        save_predictions(name, y_test, y_proba)

        if persist_models:
            config.ensure_directories()
            key = name.lower().replace(" ", "_")
            joblib.dump(model, config.RESULTS_MODELS_DIR / f"{key}.joblib")

    return models, results


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.DATA_PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    _, results = run_baseline_pipeline(X_train, y_train, X_test, y_test)
    for name, metrics in results.items():
        print(name, metrics)
