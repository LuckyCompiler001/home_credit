"""Training performance prober for the Home Credit experiment."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aim.sdk.agent.research_agent_logger import (
    AGENT_LOG_PROBER_PREFIX,
    AGENT_LOG_TEST_PREFIX,
    ResearchAgentLogger,
)
from sklearn.metrics import roc_auc_score


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
import train


DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / ".codex" / "performance_history.json"
DEFAULT_FIGURE_PATH = Path(__file__).resolve().parent / "result" / "performance_history.png"
HISTORY_MAX_LENGTH = 50
EPS = 1e-7


@dataclass(frozen=True)
class DatasetSplits:
    """Container for Home Credit train/validation arrays."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    feature_columns: list[str]
    baseline_prob: float

    @property
    def n_train_rows(self) -> int:
        return int(self.y_train.shape[0])

    @property
    def n_valid_rows(self) -> int:
        return int(self.y_valid.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1]) if self.X_train.ndim == 2 else 0


@dataclass(frozen=True)
class PerformanceMetrics:
    """Captures the metrics logged by the prober."""

    train_log_loss: float
    valid_log_loss: float
    train_auroc: float
    valid_auroc: float
    train_accuracy: float
    valid_accuracy: float
    baseline_prob: float
    n_train_rows: int
    n_valid_rows: int
    n_features: int
    timestamp: float


class TrainingPerformanceProber:
    """Re-runs lightweight evaluation and tracks Home Credit model health."""

    def __init__(
        self,
        *,
        logger: ResearchAgentLogger | None = None,
        history_path: Path | None = None,
        figure_path: Path | None = None,
    ) -> None:
        self.logger = logger or ResearchAgentLogger()
        self.history_path = Path(history_path or DEFAULT_HISTORY_PATH)
        self.figure_path = Path(figure_path or DEFAULT_FIGURE_PATH)

    def run(self) -> PerformanceMetrics:
        dataset = self._load_dataset()
        weights, train_pred, valid_pred = self._fit_model(dataset)
        metrics = self._compute_metrics(dataset, train_pred, valid_pred)
        self._log_metrics(metrics)
        history = self._persist_history(metrics)
        figure_path = self._render_history_plot(history)
        if figure_path is not None:
            self.logger.track_figure(
                f"{AGENT_LOG_PROBER_PREFIX}performance_history",
                str(figure_path),
            )
        return metrics

    def _load_dataset(self) -> DatasetSplits:
        train_frame = pd.read_csv(train.DATA_DIR / "application_train.csv")
        test_frame = pd.read_csv(train.DATA_DIR / "application_test.csv")

        y = train_frame["TARGET"].astype(float)
        train_features = train._prepare_numeric_frame(train_frame, drop_target=True)
        test_features = train._prepare_numeric_frame(test_frame, drop_target=False)

        feature_columns = train._select_feature_columns(train_features, test_features)
        if not feature_columns:
            raise RuntimeError("No overlapping numeric feature columns available")

        X = train_features[feature_columns].to_numpy(dtype=np.float32)
        y_array = y.to_numpy(dtype=np.float32)

        split_index = max(1, int(len(X) * (1 - train.VALID_FRACTION)))
        X_train = X[:split_index]
        y_train = y_array[:split_index]
        X_valid = X[split_index:]
        y_valid = y_array[split_index:]

        baseline_prob = float(y_train.mean()) if y_train.size else 0.5
        return DatasetSplits(X_train, y_train, X_valid, y_valid, feature_columns, baseline_prob)

    def _fit_model(
        self,
        dataset: DatasetSplits,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = train._train_simple_model(dataset.X_train, dataset.y_train)
        if weights.size == 0:
            baseline = dataset.baseline_prob
            train_pred = np.full(dataset.n_train_rows, baseline, dtype=np.float32)
            valid_pred = np.full(dataset.n_valid_rows, baseline, dtype=np.float32)
            return weights, train_pred, valid_pred
        train_pred = train._predict(weights, dataset.X_train) if dataset.n_train_rows else np.zeros(0, dtype=np.float32)
        valid_pred = train._predict(weights, dataset.X_valid) if dataset.n_valid_rows else np.zeros(0, dtype=np.float32)
        return weights, train_pred, valid_pred

    def _compute_metrics(
        self,
        dataset: DatasetSplits,
        train_pred: np.ndarray,
        valid_pred: np.ndarray,
    ) -> PerformanceMetrics:
        train_log_loss = self._binary_log_loss(dataset.y_train, train_pred)
        valid_log_loss = self._binary_log_loss(dataset.y_valid, valid_pred)
        train_auroc = self._safe_roc_auc(dataset.y_train, train_pred)
        valid_auroc = self._safe_roc_auc(dataset.y_valid, valid_pred)
        train_accuracy = self._classification_accuracy(dataset.y_train, train_pred)
        valid_accuracy = self._classification_accuracy(dataset.y_valid, valid_pred)
        timestamp = time.time()
        return PerformanceMetrics(
            train_log_loss=train_log_loss,
            valid_log_loss=valid_log_loss,
            train_auroc=train_auroc,
            valid_auroc=valid_auroc,
            train_accuracy=train_accuracy,
            valid_accuracy=valid_accuracy,
            baseline_prob=dataset.baseline_prob,
            n_train_rows=dataset.n_train_rows,
            n_valid_rows=dataset.n_valid_rows,
            n_features=dataset.n_features,
            timestamp=timestamp,
        )

    def _binary_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        clipped = np.clip(y_pred.astype(np.float64), EPS, 1.0 - EPS)
        targets = y_true.astype(np.float64)
        loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        return float(loss)

    def _safe_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.5
        if np.unique(y_true).size < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return 0.5

    def _classification_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        labels = (y_pred >= 0.5).astype(np.float32)
        return float((labels == y_true).mean())

    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}train_log_loss",
            metrics.train_log_loss,
            split="train",
            metric="log_loss",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}valid_log_loss",
            metrics.valid_log_loss,
            split="validation",
            metric="log_loss",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}train_auroc",
            metrics.train_auroc,
            split="train",
            metric="roc_auc",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}valid_auroc",
            metrics.valid_auroc,
            split="validation",
            metric="roc_auc",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}train_accuracy",
            metrics.train_accuracy,
            split="train",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}valid_accuracy",
            metrics.valid_accuracy,
            split="validation",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}baseline_probability",
            metrics.baseline_prob,
            split="train",
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}n_features",
            metrics.n_features,
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}n_train_rows",
            metrics.n_train_rows,
        )
        self.logger.log(
            f"{AGENT_LOG_TEST_PREFIX}n_valid_rows",
            metrics.n_valid_rows,
        )
        loss_gap = metrics.valid_log_loss - metrics.train_log_loss
        auc_gap = metrics.valid_auroc - metrics.train_auroc
        self.logger.log(
            f"{AGENT_LOG_PROBER_PREFIX}generalization_gap_log_loss",
            loss_gap,
        )
        self.logger.log(
            f"{AGENT_LOG_PROBER_PREFIX}generalization_gap_auroc",
            auc_gap,
        )
        self.logger.log(
            f"{AGENT_LOG_PROBER_PREFIX}training_hyperparameters",
            1.0,
            learning_rate=train.LEARNING_RATE,
            num_epochs=train.NUM_EPOCHS,
            max_feature_columns=train.MAX_FEATURE_COLUMNS,
        )

    def _persist_history(self, metrics: PerformanceMetrics) -> list[dict[str, Any]]:
        history = self._load_history()
        entry = asdict(metrics)
        history.append(entry)
        trimmed = history[-HISTORY_MAX_LENGTH:]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(trimmed, handle, indent=2)
        return trimmed

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                cleaned: list[dict[str, Any]] = []
                for entry in data:
                    if isinstance(entry, dict):
                        cleaned.append(entry)
                return cleaned
        except (OSError, json.JSONDecodeError):
            return []
        return []

    def _render_history_plot(self, history: list[dict[str, Any]]) -> Path | None:
        if not history:
            return None
        x_values = list(range(1, len(history) + 1))
        train_auc = [float(item.get("train_auroc", 0.0)) for item in history]
        valid_auc = [float(item.get("valid_auroc", 0.0)) for item in history]
        train_loss = [float(item.get("train_log_loss", 0.0)) for item in history]
        valid_loss = [float(item.get("valid_log_loss", 0.0)) for item in history]

        fig, (ax_auc, ax_loss) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        ax_auc.plot(x_values, train_auc, marker="o", label="Train AUROC", color="#2a6fbb")
        ax_auc.plot(x_values, valid_auc, marker="o", label="Validation AUROC", color="#c44e52")
        ax_auc.set_ylabel("AUROC")
        ax_auc.set_ylim(0.0, 1.0)
        ax_auc.set_title("Home Credit Performance History")
        ax_auc.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax_auc.legend(loc="lower right")

        ax_loss.plot(x_values, train_loss, marker="o", label="Train Log Loss", color="#55a868")
        ax_loss.plot(x_values, valid_loss, marker="o", label="Validation Log Loss", color="#dd8452")
        ax_loss.set_xlabel("Prober run")
        ax_loss.set_ylabel("Binary Log Loss")
        max_loss = max(train_loss + valid_loss) if (train_loss or valid_loss) else 1.0
        ax_loss.set_ylim(0.0, max(1.0, max_loss * 1.15))
        ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax_loss.legend(loc="best")

        ax_loss.set_xticks(x_values)
        fig.tight_layout()

        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_path, bbox_inches="tight")
        plt.close(fig)
        return self.figure_path


__all__ = ["TrainingPerformanceProber", "PerformanceMetrics"]
