"""Phase 4: RXT — ResNeXt-embedded GRU, the project's core contribution.

Based on the architectural idea in Almazroi & Ayub (2023): grouped ("cardinality")
convolutions extract local feature interactions, then a GRU learns dependencies
across the resulting feature sequence, before a dense sigmoid head classifies.

Design decisions (documented here for Chapter 4.4 / viva defence):

- Input reshape: the 30 tabular features (V1-V28, Time, Amount) have no native
  sequential order. They are reshaped to (30, 1) - a length-30 "sequence" with
  1 channel - purely so 1D grouped convolutions have something to slide over.
  This is a deliberate repurposing of a sequence architecture for tabular data,
  not a claim that the features are temporally ordered - that repurposing is
  exactly what Almazroi & Ayub's RXT design does, and it is worth defending
  explicitly rather than glossing over.
- Cardinality (grouped convolutions): splitting each conv's channels into
  `cardinality` independent groups (Xie et al., 2017's "ResNeXt" dimension)
  gives the block more representational diversity per parameter than a single
  wide convolution would, at a fraction of the parameter cost of widening the
  layer outright. Implemented directly via Keras Conv1D's `groups` argument.
- Residual/skip connections around each block combat vanishing gradients and
  let the network default to an identity mapping if a block isn't useful -
  standard ResNet motivation, carried into the "ResNeXt" naming.
- GRU (not LSTM): fewer gates than LSTM (no separate cell state), so fewer
  parameters and faster training, while still handling the vanishing-gradient
  problem that plain RNNs suffer from - a reasonable efficiency/capability
  trade-off given Phase 7 explicitly benchmarks computational cost.
- Dropout + L2 + early stopping + k-fold CV: all four are direct responses to
  the ToR's explicitly flagged overfitting risk for this architecture.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src import config
from src.evaluate import compute_metrics, evaluate_model, save_predictions, select_threshold

tf.random.set_seed(config.RANDOM_SEED)


def reshape_for_rxt(X) -> np.ndarray:
    """(n_samples, n_features) -> (n_samples, n_features, 1) for Conv1D/GRU input."""
    X_array = X.values if hasattr(X, "values") else np.asarray(X)
    return X_array.reshape(X_array.shape[0], X_array.shape[1], 1)


def resnext_block(x, filters: int, cardinality: int = 8, kernel_size: int = 3, l2_reg: float = 1e-4):
    """One ResNeXt-style block: 1x1 reduce -> grouped conv -> 1x1 expand, with
    a residual connection. BatchNorm + ReLU after each conv per the standard
    ResNet(-eXt) pre-activation-free design.
    """
    shortcut = x

    y = layers.Conv1D(filters, kernel_size=1, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        groups=cardinality,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(filters, kernel_size=1, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(y)
    y = layers.BatchNormalization()(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([shortcut, y])
    out = layers.ReLU()(out)
    return out


def build_rxt_model(
    n_features: int,
    cardinality: int = 8,
    gru_units: int = 64,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Assemble the full RXT model: two ResNeXt blocks -> GRU -> dropout -> sigmoid head."""
    inputs = layers.Input(shape=(n_features, 1), name="tabular_sequence_input")

    x = resnext_block(inputs, filters=32, cardinality=cardinality, l2_reg=l2_reg)
    x = resnext_block(x, filters=64, cardinality=cardinality, l2_reg=l2_reg)

    x = layers.GRU(gru_units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_reg))(x)

    model = keras.Model(inputs, outputs, name="RXT_ResNeXt_GRU")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_rxt(
    X_train,
    y_train,
    X_val,
    y_val,
    class_weight: dict | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    cardinality: int = 8,
    gru_units: int = 64,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    verbose: int = 0,
):
    """Train one RXT model with early stopping on validation loss."""
    model = build_rxt_model(
        n_features=X_train.shape[1],
        cardinality=cardinality,
        gru_units=gru_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        reshape_for_rxt(X_train),
        y_train,
        validation_data=(reshape_for_rxt(X_val), y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return model, history


def train_rxt_kfold(
    X,
    y,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    class_weight: dict | None = None,
    verbose: int = 0,
) -> "pd.DataFrame":
    """k-fold cross-validation for RXT, per the ToR's overfitting-mitigation plan.

    Each fold trains a fresh model (no weight sharing across folds) and is
    evaluated through evaluate.compute_metrics - the same function used for
    every baseline - so fold results are directly comparable to Phase 3.

    Threshold handling here is deliberately looser than run_rxt_pipeline()
    below: each fold's held-out partition serves double duty as both the
    early-stopping validation set and the data the fold is scored on, so
    tuning the threshold on it too is a mild form of the same leakage the
    three-way split elsewhere in this project exists to avoid. That's an
    accepted simplification for this specific function because k-fold CV here
    is a supplementary robustness check (Chapter 6.2's mean+/-std), not the
    headline reported result - the properly separated val/test evaluation in
    run_rxt_pipeline() is what belongs in the main model-comparison table.
    """
    import pandas as pd

    X_array = X.values if hasattr(X, "values") else np.asarray(X)
    y_array = y.values if hasattr(y, "values") else np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array)):
        X_fold_train, X_fold_val = X_array[train_idx], X_array[val_idx]
        y_fold_train, y_fold_val = y_array[train_idx], y_array[val_idx]

        model, _ = train_rxt(
            X_fold_train,
            y_fold_train,
            X_fold_val,
            y_fold_val,
            class_weight=class_weight,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        y_proba = model.predict(reshape_for_rxt(X_fold_val), verbose=0).flatten()
        threshold = select_threshold(y_fold_val, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_metrics(y_fold_val, y_pred, y_proba)
        metrics["threshold"] = threshold
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    results_df = pd.DataFrame(fold_results).set_index("fold")
    return results_df


def run_rxt_pipeline(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_weight: dict | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    persist_model: bool = True,
    verbose: int = 0,
):
    """Mirrors baseline_models.run_baseline_pipeline: train RXT using the
    project's real validation split (for early stopping AND threshold tuning
    - see evaluate.select_threshold), then evaluate once, at that one fixed
    threshold, on the untouched held-out test set via the same evaluate.py
    pipeline used for every baseline. Persists the model + predictions.

    An earlier version of this function carved its own throwaway validation
    split out of X_train and used a hardcoded 0.5 threshold. Now that
    preprocessing.load_and_split() already produces a proper three-way
    train/val/test split, reusing that validation set here - for both early
    stopping and threshold selection - avoids a redundant extra split and
    keeps every model (baselines and RXT alike) tuned against the same
    validation data.

    This is the RXT result that belongs in the main model-comparison table
    (Chapter 6.3) - directly comparable to the baselines because it is tuned
    and evaluated the same way, on the same splits. train_rxt_kfold() above is
    a separate, supplementary robustness check (Chapter 6.2), not a
    replacement for this held-out evaluation.
    """
    model, history = train_rxt(
        X_train, y_train, X_val, y_val,
        class_weight=class_weight, epochs=epochs, batch_size=batch_size, verbose=verbose,
    )

    y_proba_val = model.predict(reshape_for_rxt(X_val), verbose=0).flatten()
    threshold = select_threshold(y_val, y_proba_val)

    y_proba_test = model.predict(reshape_for_rxt(X_test), verbose=0).flatten()
    y_pred_test = (y_proba_test >= threshold).astype(int)

    metrics = evaluate_model("RXT (ResNeXt-GRU)", y_test, y_pred_test, y_proba_test, threshold=threshold)
    save_predictions("RXT (ResNeXt-GRU)", y_test, y_proba_test)

    if persist_model:
        config.ensure_directories()
        model.save(config.RESULTS_MODELS_DIR / "rxt_resnext_gru.keras")

    return model, metrics, history


def summarize_kfold_results(results_df) -> "pd.DataFrame":
    """Mean +/- std per metric across folds - the format Chapter 6 reports RXT in."""
    import pandas as pd

    summary = results_df.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return summary


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(config.DATA_PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(config.DATA_PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_val = val_df.drop(columns=[config.TARGET_COLUMN])
    y_val = val_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    from src.preprocessing import compute_class_weights

    weights = compute_class_weights(y_train)

    _, metrics, _ = run_rxt_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test, class_weight=weights, epochs=50
    )
    print("Held-out test set metrics:", metrics)

    kfold_results = train_rxt_kfold(X_train, y_train, n_splits=5, epochs=30, class_weight=weights)
    print(kfold_results)
    summary = summarize_kfold_results(kfold_results)
    print(summary)
    summary.to_csv(config.RESULTS_METRICS_DIR / "rxt_kfold_summary.csv")
