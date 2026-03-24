"""
Classification model for macro regime prediction.

Two modes:
  - "logistic":     Full retrain each step (LogisticRegression)
  - "incremental":  Online learning via SGDClassifier with partial_fit
                    Model carries forward between steps and updates on new data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from config import Config


class RegimeClassifier:
    """Wraps a scikit-learn classifier for binary regime classification."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.classifier = self._build_classifier()
        self.feature_names: list = []
        self._is_fitted = False

    def _build_classifier(self):
        if self.cfg.model_type == "logistic":
            return LogisticRegression(
                C=self.cfg.regularization_C,
                class_weight=self.cfg.class_weight,
                max_iter=self.cfg.max_iter,
                solver="lbfgs",
                random_state=42,
            )
        elif self.cfg.model_type == "incremental":
            # class_weight="balanced" doesn't work with partial_fit on
            # single-class batches, so we leave it None and handle
            # weighting via sample_weight in the caller instead.
            return SGDClassifier(
                loss="log_loss",
                alpha=self.cfg.sgd_alpha,
                class_weight=None,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                warm_start=True,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.cfg.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None):
        """Full train on feature matrix X and labels y."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.classifier.fit(X_scaled, y.values, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def partial_fit(self, X: pd.DataFrame, y: pd.Series,
                    sample_weight: np.ndarray = None):
        """
        Incremental update. Only available for SGDClassifier.
        On first call, does initial fit. After that, updates existing weights.
        """
        if self.cfg.model_type != "incremental":
            raise RuntimeError("partial_fit only available with model_type='incremental'")

        self.feature_names = list(X.columns)
        classes = np.array([0, 1])

        if not self._is_fitted:
            # First call: fit the scaler on initial batch
            X_scaled = self.scaler.fit_transform(X.values)
            self.classifier.partial_fit(X_scaled, y.values, classes=classes,
                                        sample_weight=sample_weight)
            self._is_fitted = True
        else:
            # Keep feature scaling current as the data distribution evolves.
            self.scaler.partial_fit(X.values)
            X_scaled = self.scaler.transform(X.values)
            self.classifier.partial_fit(X_scaled, y.values,
                                        sample_weight=sample_weight)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X.values)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities. Shape: (n_samples, 2)."""
        X_scaled = self.scaler.transform(X.values)
        return self.classifier.predict_proba(X_scaled)

    def get_coefficients(self) -> pd.DataFrame:
        """Extract model coefficients. Rows = classes, columns = features."""
        coefs = self.classifier.coef_
        class_names = [self.cfg.class_labels[i] for i in range(coefs.shape[0])]
        return pd.DataFrame(coefs, index=class_names, columns=self.feature_names)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def save_model(self, path: str = None):
        path = path or self.cfg.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }, path)
        print(f"  Model saved to {path}")

    def load_model(self, path: str = None):
        path = path or self.cfg.model_path
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.classifier = data["classifier"]
        self.feature_names = data["feature_names"]
        self._is_fitted = data["is_fitted"]
        print(f"  Model loaded from {path}")
        return self

    def save_checkpoint(self, step: int, cfg: Config):
        """Save a timestamped checkpoint."""
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, f"model_step_{step:04d}.joblib")
        self.save_model(path)
