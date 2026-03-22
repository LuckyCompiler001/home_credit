"""Train a Home Credit default-risk baseline model.

This script follows the README task definition:
- Binary classification (`TARGET`)
- ROC-AUC validation metric
- LightGBM baseline on `application_train.csv`

Run:
    python3 train.py
"""

from __future__ import annotations

import json
import logging
import re
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = Path('/home/xuanhe_linux_001/aim_frontend_experiment3/aim/examples/agent_example_repos/home_credit/result')
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 3000
LEARNING_RATE = 0.03
NUM_LEAVES = 64
N_JOBS = -1
INCLUDE_AUXILIARY = False


def sanitize_days_columns(frame: pd.DataFrame) -> pd.DataFrame:
    day_cols = [column for column in frame.columns if column.startswith('DAYS_')]
    for column in day_cols:
        frame[column] = frame[column].replace(365243, np.nan)
    return frame


def aggregate_numeric_table(table: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    if key not in table.columns:
        raise ValueError(f'Missing group key `{key}` in table `{prefix}`')

    table = sanitize_days_columns(table.copy())
    numeric_columns = [
        column
        for column in table.select_dtypes(include=[np.number]).columns
        if column != key
    ]

    if not numeric_columns:
        grouped = table[[key]].drop_duplicates().set_index(key)
    else:
        grouped = table.groupby(key)[numeric_columns].agg(['mean', 'max', 'min', 'sum'])
        grouped.columns = [f'{prefix}_{column}_{stat}'.upper() for column, stat in grouped.columns]

    grouped[f'{prefix}_ROW_COUNT'.upper()] = table.groupby(key).size()
    return grouped


def build_auxiliary_features(data_dir: Path) -> dict[str, pd.DataFrame]:
    logger.info('Building auxiliary aggregated features ...')
    features: dict[str, pd.DataFrame] = {}

    previous_application = pd.read_csv(data_dir / 'previous_application.csv')
    features['PREV'] = aggregate_numeric_table(previous_application, key='SK_ID_CURR', prefix='PREV')

    pos_cash_balance = pd.read_csv(data_dir / 'POS_CASH_balance.csv')
    features['POS_CASH'] = aggregate_numeric_table(pos_cash_balance, key='SK_ID_CURR', prefix='POS')

    installments = pd.read_csv(data_dir / 'installments_payments.csv')
    features['INSTAL'] = aggregate_numeric_table(installments, key='SK_ID_CURR', prefix='INSTAL')

    credit_card = pd.read_csv(data_dir / 'credit_card_balance.csv')
    features['CC'] = aggregate_numeric_table(credit_card, key='SK_ID_CURR', prefix='CC')

    bureau = pd.read_csv(data_dir / 'bureau.csv')
    bureau_balance = pd.read_csv(data_dir / 'bureau_balance.csv')
    bureau_balance_agg = aggregate_numeric_table(bureau_balance, key='SK_ID_BUREAU', prefix='BB')
    bureau_with_balance = bureau.merge(bureau_balance_agg, left_on='SK_ID_BUREAU', right_index=True, how='left')
    features['BUREAU'] = aggregate_numeric_table(bureau_with_balance, key='SK_ID_CURR', prefix='BUREAU')

    return features


def encode_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.concat([train_frame, test_frame], axis=0, ignore_index=True)
    merged = sanitize_days_columns(merged)
    categorical_columns = [
        column
        for column in merged.columns
        if pd.api.types.is_object_dtype(merged[column].dtype)
        or pd.api.types.is_string_dtype(merged[column].dtype)
    ]
    merged = pd.get_dummies(merged, columns=categorical_columns, dummy_na=True)
    merged.columns = sanitize_feature_names(merged.columns)
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    train_encoded = merged.iloc[: len(train_frame)].copy()
    test_encoded = merged.iloc[len(train_frame) :].copy()
    return train_encoded, test_encoded


def sanitize_feature_names(columns: pd.Index) -> list[str]:
    sanitized_columns: list[str] = []
    seen_names: dict[str, int] = {}

    for column in columns:
        normalized = re.sub(r'[^0-9A-Za-z_]+', '_', str(column)).strip('_')
        if not normalized:
            normalized = 'feature'
        if normalized[0].isdigit():
            normalized = f'F_{normalized}'

        duplicate_count = seen_names.get(normalized, 0)
        seen_names[normalized] = duplicate_count + 1
        if duplicate_count:
            normalized = f'{normalized}_{duplicate_count}'

        sanitized_columns.append(normalized)

    return sanitized_columns


def attach_auxiliary_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    auxiliary: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for feature_name, feature_frame in auxiliary.items():
        logger.info('Joining %s features (%d columns)', feature_name, feature_frame.shape[1])
        train_frame = train_frame.merge(feature_frame, left_on='SK_ID_CURR', right_index=True, how='left')
        test_frame = test_frame.merge(feature_frame, left_on='SK_ID_CURR', right_index=True, how='left')
    return train_frame, test_frame


def train() -> None:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency `lightgbm`. Install required packages first: '
            '`pip install lightgbm pandas scikit-learn`'
        ) from exc

    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / 'application_train.csv'
    test_path = data_dir / 'application_test.csv'
    sample_submission_path = data_dir / 'sample_submission.csv'

    logger.info('Loading application tables ...')
    train_application = pd.read_csv(train_path)
    test_application = pd.read_csv(test_path)

    y = train_application['TARGET'].astype(int)
    train_ids = train_application['SK_ID_CURR'].copy()
    test_ids = test_application['SK_ID_CURR'].copy()

    train_features = train_application.drop(columns=['TARGET'])
    test_features = test_application.copy()

    if INCLUDE_AUXILIARY:
        auxiliary = build_auxiliary_features(data_dir)
        train_features, test_features = attach_auxiliary_features(train_features, test_features, auxiliary)

    logger.info('Encoding features ...')
    train_features, test_features = encode_features(train_features, test_features)

    X = train_features.drop(columns=['SK_ID_CURR'], errors='ignore')
    X_test = test_features.drop(columns=['SK_ID_CURR'], errors='ignore')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pos_count = int(y_train.sum())
    neg_count = int((1 - y_train).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    logger.info('Training LightGBM baseline ...')
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)],
    )

    val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = float(roc_auc_score(y_valid, val_pred))
    logger.info('Validation ROC-AUC: %.6f', val_auc)

    test_pred = model.predict_proba(X_test)[:, 1]

    if sample_submission_path.exists():
        submission = pd.read_csv(sample_submission_path)
        if 'SK_ID_CURR' not in submission.columns:
            submission.insert(0, 'SK_ID_CURR', test_ids.values)
        submission['TARGET'] = test_pred
    else:
        submission = pd.DataFrame({'SK_ID_CURR': test_ids.values, 'TARGET': test_pred})

    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics_path = output_dir / 'validation_metrics.json'
    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'include_auxiliary': bool(INCLUDE_AUXILIARY),
    }
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    feature_importance = pd.DataFrame(
        {
            'feature': X.columns,
            'importance_gain': model.booster_.feature_importance(importance_type='gain'),
            'importance_split': model.booster_.feature_importance(importance_type='split'),
        }
    ).sort_values('importance_gain', ascending=False)
    feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)

    logger.info('Saved submission: %s', submission_path)
    logger.info('Saved metrics: %s', metrics_path)

def main() -> None:
    train()


if __name__ == '__main__':
    main()
