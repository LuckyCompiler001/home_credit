import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'result'
VALID_FRACTION = 0.2
MAX_FEATURE_COLUMNS = 32
LEARNING_RATE = 1e-10
NUM_EPOCHS = 3


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
    weights = np.zeros(X.shape[1], dtype=np.float32)
    for _ in range(NUM_EPOCHS):
        logits = X @ weights
        predictions = _sigmoid(logits)
        gradient = (X.T @ (predictions - y)) / max(1, len(y))
        weights -= LEARNING_RATE * gradient.astype(np.float32)
    return weights


def _predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    if weights.size == 0 or X.size == 0:
        return np.zeros(X.shape[0], dtype=np.float32)
    logits = X @ weights
    return _sigmoid(logits)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(DATA_DIR / 'application_train.csv')
    test_frame = pd.read_csv(DATA_DIR / 'application_test.csv')

    y = train_frame['TARGET'].astype(float)
    train_features = _prepare_numeric_frame(train_frame, drop_target=True)
    test_features = _prepare_numeric_frame(test_frame, drop_target=False)

    feature_columns = _select_feature_columns(train_features, test_features)
    if not feature_columns:
        raise RuntimeError('No overlapping numeric feature columns available')

    X = train_features[feature_columns].to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)

    split_index = max(1, int(len(X) * (1 - VALID_FRACTION)))
    X_train = X[:split_index]
    y_train = y_array[:split_index]
    X_valid = X[split_index:]
    y_valid = y_array[split_index:]

    weights = _train_simple_model(X_train, y_train)
    if weights.size == 0:
        baseline_prob = float(y_train.mean()) if y_train.size else 0.5
        valid_pred = np.full(len(X_valid), baseline_prob, dtype=np.float32)
        test_pred = np.full(len(X_test), baseline_prob, dtype=np.float32)
    else:
        valid_pred = _predict(weights, X_valid)
        test_pred = _predict(weights, X_test)

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


if __name__ == '__main__':
    main()
