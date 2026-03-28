import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'result'
VALID_FRACTION = 0.2


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / 'application_train.csv'
    test_path = DATA_DIR / 'application_test.csv'

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)

    # Only keep numeric columns and replace everything missing with zeros.
    train_numeric = train_frame.select_dtypes(include=[np.number]).fillna(0.0)
    test_numeric = test_frame.select_dtypes(include=[np.number]).fillna(0.0)

    y = train_numeric['TARGET'].astype(float)
    features = train_numeric.drop(columns=['TARGET'], errors='ignore')
    shared_columns = [column for column in features.columns if column in test_numeric.columns]
    if not shared_columns:
        raise RuntimeError('No shared numeric columns between train and test frames')

    X = features[shared_columns]
    X_test = test_numeric[shared_columns]

    split_index = max(1, int(len(X) * (1 - VALID_FRACTION)))
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_valid = X.iloc[split_index:]
    y_valid = y.iloc[split_index:]

    baseline_prob = float(y_train.mean()) if not y_train.empty else 0.5

    train_pred = np.full(len(X_train), baseline_prob, dtype=float)
    valid_pred = np.full(len(X_valid), baseline_prob, dtype=float)

    if len(y_valid):
        val_auc = float(roc_auc_score(y_valid, valid_pred))
    else:
        val_auc = 0.5

    test_pred = np.full(len(X_test), baseline_prob, dtype=float)

    submission = pd.DataFrame({'SK_ID_CURR': test_frame['SK_ID_CURR'], 'TARGET': test_pred})
    submission_path = RESULT_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'train_baseline_probability': baseline_prob,
    }

    metrics_path = RESULT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    print('Saved submission to', submission_path)
    print('Validation ROC-AUC:', val_auc)


if __name__ == '__main__':
    main()
