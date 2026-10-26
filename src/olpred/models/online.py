"""Online (sequential) learning: weights adjusted as results come in.

``OnlineSoftmax`` is a multinomial logistic regression trained by plain
gradient descent on the log loss. It starts from a fitted ``LogisticModel`` and,
after each batch of finished matches, takes gradient steps on that batch:

    P = softmax(X W^T + b)            (predict before the matches are played)
    ∇W = (P − Y)^T X / n + λ W        (update once the results are known)
    W ← W − η ∇W

The L2 term λ = 1 / (C · N_train) is the same regularisation as the batch model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from olpred.models.baselines import LogisticModel


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OnlineSoftmax:
    def __init__(self, base: LogisticModel, n_train: int, lr: float, epochs: int = 1):
        self.cols = base.cols
        self.prep = base.pipe_[:-1]  # imputer + scaler fitted on the training seasons only
        clf = base.pipe_[-1]
        self.W, self.b = clf.coef_.copy(), clf.intercept_.copy()
        self.lr, self.epochs = lr, epochs
        self.l2 = 1.0 / (base.C_ * n_train)

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return self.prep.transform(df[self.cols])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return softmax(self._x(df) @ self.W.T + self.b)

    def update(self, df: pd.DataFrame) -> None:
        X, Y = self._x(df), np.eye(3)[df["y"].to_numpy()]
        for _ in range(self.epochs):
            G = softmax(X @ self.W.T + self.b) - Y
            self.W -= self.lr * (G.T @ X / len(df) + self.l2 * self.W)
            self.b -= self.lr * G.mean(axis=0)


def refit_logistic(cols: list[str], C: float, df: pd.DataFrame):
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=500))
    return pipe.fit(df[cols], df["y"])


def week_ids(dates: pd.Series) -> np.ndarray:
    return ((dates - dates.min()).dt.days // 7).to_numpy()


def prequential(base: LogisticModel, train: pd.DataFrame, stream: pd.DataFrame, lr: float,
                epochs: int = 1, with_refit: bool = True) -> dict[str, np.ndarray]:
    """Walk through ``stream`` week by week. Each week is predicted *first* with the
    information available so far, then its results are revealed and used to update.

    Returns per-match probabilities for: static (never updated), online (gradient
    updates) and, optionally, refit (full re-estimation every week)."""
    online = OnlineSoftmax(base, len(train), lr, epochs)
    weeks = week_ids(stream["date"])
    out = {"static": [], "online": []}
    if with_refit:
        out["refit"] = []
    seen = train
    for w in np.unique(weeks):
        batch = stream[weeks == w]
        out["static"].append(base.predict_proba(batch))
        out["online"].append(online.predict_proba(batch))
        if with_refit:
            out["refit"].append(refit_logistic(base.cols, base.C_, seen).predict_proba(batch[base.cols]))
        online.update(batch)
        seen = pd.concat([seen, batch], ignore_index=True)
    return {k: np.vstack(v) for k, v in out.items()}
