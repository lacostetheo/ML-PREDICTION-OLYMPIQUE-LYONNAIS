import numpy as np
import pandas as pd

from olpred.models.baselines import LogisticModel
from olpred.models.online import OnlineSoftmax, softmax


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    logits = np.column_stack([-x[:, 0], 0.2 * np.ones(n), x[:, 0]]) + 0.3 * rng.normal(size=(n, 3))
    y = np.array([rng.choice(3, p=p) for p in softmax(logits)])
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df["y"] = y
    return df


def test_online_model_starts_identical_to_base_model():
    df = _data()
    base = LogisticModel(["a", "b", "c"]).fit(df)
    online = OnlineSoftmax(base, len(df), lr=0.1)
    assert np.allclose(online.predict_proba(df), base.predict_proba(df), atol=1e-8)


def test_update_reduces_loss_on_the_batch_it_sees():
    df = _data()
    base = LogisticModel(["a", "b", "c"]).fit(df.iloc[:100])
    online = OnlineSoftmax(base, 100, lr=0.5, epochs=5)
    batch = df.iloc[100:400]

    def ll(p):
        return -np.log(p[np.arange(len(batch)), batch["y"].to_numpy()]).mean()

    before = ll(online.predict_proba(batch))
    online.update(batch)
    assert ll(online.predict_proba(batch)) < before


def test_probabilities_stay_valid_after_updates():
    df = _data()
    base = LogisticModel(["a", "b", "c"]).fit(df.iloc[:200])
    online = OnlineSoftmax(base, 200, lr=0.3)
    for i in range(200, 600, 40):
        online.update(df.iloc[i:i + 40])
    P = online.predict_proba(df)
    assert np.allclose(P.sum(axis=1), 1) and (P > 0).all()
