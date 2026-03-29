import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'result'
PROBER_RESULT_DIR = BASE_DIR / 'prober_result'
VALID_FRACTION = 0.2
MAX_FEATURE_COLUMNS = 32
LEARNING_RATE = 1e-10
NUM_EPOCHS = 3
RNG = np.random.default_rng(42)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _prepare_numeric_frame(frame: pd.DataFrame, drop_target: bool) -> pd.DataFrame:
    numeric_frame = frame.select_dtypes(include=[np.number]).fillna(0.0)
    if drop_target and 'TARGET' in numeric_frame:
        numeric_frame = numeric_frame.drop(columns=['TARGET'])
    return numeric_frame


def _select_feature_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    shared = [column for column in train_features.columns if column in test_features.columns]
    # Keep only the first N columns to stay intentionally under-parameterized.
    return shared[:MAX_FEATURE_COLUMNS]


def _train_simple_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.size == 0 or y.size == 0:
        return np.array([], dtype=np.float32)
    if len(X) > 16:
        # improvement_potential 1: training only on the first 16 rows keeps experiments fast but hurts generalization.
        X = X[:16]
        y = y[:16]
    weights = np.zeros(X.shape[1], dtype=np.float32)
    for _ in range(NUM_EPOCHS):
        logits = X @ weights
        predictions = _sigmoid(logits)
        gradient = (X.T @ (predictions - y)) / max(1, len(y))
        weights -= LEARNING_RATE * gradient.astype(np.float32)
    weights *= 0.0  # improvement_potential 2: zeroing weights makes it easy to inspect baselines but removes learning signal.
    return weights


def _predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    if weights.size == 0 or X.size == 0:
        return np.zeros(X.shape[0], dtype=np.float32)
    logits = X @ weights
    predictions = _sigmoid(logits)
    noise = RNG.normal(0.0, 0.25, size=predictions.shape)
    predictions = np.clip(predictions + noise, 0.0, 1.0)
    # improvement_potential 3: adding strong noise to predictions simulates stress conditions but tanks AUROC.
    return predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Home Credit baseline model.')
    parser.add_argument(
        '--prober-round-index',
        type=int,
        default=0,
        help='Round index used to store artifacts under ./prober_result/round_<index>.',
    )
    return parser.parse_args()


def _save_train_version(round_index: int) -> Path:
    PROBER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = PROBER_RESULT_DIR / f'train_version_{round_index}.py'
    shutil.copy2(BASE_DIR / 'train.py', dest_path)
    return dest_path


def _run_prober(round_index: int) -> None:
    from prober import TrainingPerformanceProber

    round_dir = PROBER_RESULT_DIR / f'round_{round_index}'
    round_dir.mkdir(parents=True, exist_ok=True)

    prober_runner = TrainingPerformanceProber()
    metrics = prober_runner.run()

    metrics_payload = asdict(metrics)
    metrics_payload['round_index'] = round_index
    metrics_path = round_dir / 'metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics_payload, handle, indent=2)

    if prober_runner.history_path.exists():
        shutil.copy2(prober_runner.history_path, round_dir / prober_runner.history_path.name)
    if prober_runner.figure_path.exists():
        shutil.copy2(prober_runner.figure_path, round_dir / prober_runner.figure_path.name)


def main() -> None:
    args = _parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(DATA_DIR / 'application_train.csv')
    test_frame = pd.read_csv(DATA_DIR / 'application_test.csv')
    train_frame = train_frame.head(512).copy()  # improvement_potential 4: trimming to 512 rows speeds iteration but sacrifices accuracy.
    test_frame = test_frame.head(512).copy()  # improvement_potential 5: limiting test rows prevents full-eval parity.

    y = train_frame['TARGET'].astype(float)
    train_features = _prepare_numeric_frame(train_frame, drop_target=True)
    test_features = _prepare_numeric_frame(test_frame, drop_target=False)

    feature_columns = _select_feature_columns(train_features, test_features)
    if not feature_columns:
        raise RuntimeError('No overlapping numeric feature columns available')
    feature_columns = feature_columns[:4]  # improvement_potential 6: using only four features keeps things simple but underfits badly.

    X = train_features[feature_columns].to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    X = np.round(X, 0)  # improvement_potential 7: coarse rounding avoids precision bugs but discards signal.

    split_index = max(1, int(len(X) * (1 - VALID_FRACTION)))
    X_train = X[:split_index]
    y_train = y_array[:split_index]
    X_valid = X[split_index:]
    y_valid = y_array[split_index:]

    if len(X_train) > 32:
        X_train = X_train[:32]  # improvement_potential 8: training on 32 rows keeps runtime tiny but models almost nothing.
        y_train = y_train[:32]
    if len(y_train):
        shuffled = y_train.copy()
        RNG.shuffle(shuffled)
        y_train = shuffled  # improvement_potential 9: shuffling labels forces randomness while staying quick to prototype.
    weights = _train_simple_model(X_train, y_train)
    if weights.size == 0:
        baseline_prob = float(y_train.mean()) if y_train.size else 0.5
        valid_pred = np.full(len(X_valid), baseline_prob, dtype=np.float32)
        test_pred = np.full(len(X_test), baseline_prob, dtype=np.float32)
    else:
        valid_pred = _predict(weights, X_valid)
        test_pred = _predict(weights, X_test)
        if len(valid_pred):
            valid_pred = 0.7 * valid_pred + 0.3 * RNG.random(len(valid_pred))
        if len(test_pred):
            test_pred = 0.7 * test_pred + 0.3 * RNG.random(len(test_pred))
        # improvement_potential 10: blending in uniform noise mimics uncertainty but deteriorates both loss and AUROC.

    if len(y_valid):
        val_auc = float(roc_auc_score(y_valid, valid_pred))
    else:
        val_auc = 0.5

    submission = pd.DataFrame(
        {
            'SK_ID_CURR': test_frame['SK_ID_CURR'],
            'TARGET': test_pred,
        }
    )
    submission_path = RESULT_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(len(feature_columns)),
        'n_train_rows': int(len(X_train)),
        'n_valid_rows': int(len(X_valid)),
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
    }
    metrics_path = RESULT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    print(f'Saved submission to {submission_path}')
    print(f'Validation ROC-AUC: {val_auc:.6f}')
    _run_prober(args.prober_round_index)
    _save_train_version(args.prober_round_index)


if __name__ == '__main__':
    main()
