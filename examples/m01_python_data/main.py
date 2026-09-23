"""m01 — Python & data tooling: NumPy vectorization, pandas wrangling, sklearn pipeline.

Proves theory doc 01-python-and-data-tooling.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402


def main():
    # ---- NumPy: vectorize instead of looping ----
    X = np.random.default_rng(0).normal(size=(1000, 4))
    z_scores = (X - X.mean(0)) / X.std(0)          # broadcasting
    assert z_scores.shape == (1000, 4)
    assert np.allclose(z_scores.mean(0), 0, atol=1e-6)

    # ---- pandas: build, filter, groupby, merge, missing values ----
    df = pd.DataFrame({
        "user": np.arange(20),
        "city": np.random.default_rng(1).choice(["Paris", "Tokyo", "Lima"], 20),
        "sales": np.random.default_rng(2).uniform(10, 100, 20),
    })
    df.loc[df.index[:3], "sales"] = np.nan            # inject missing
    df["sales"] = df["sales"].fillna(df["sales"].median())
    top_cities = (df.groupby("city")["sales"].mean()
                    .sort_values(ascending=False))
    merged = df.merge(pd.DataFrame({"city": top_cities.index,
                                    "rank": range(1, 4)}), on="city", how="left")
    assert merged["rank"].notna().all() and df["sales"].isna().sum() == 0

    # ---- sklearn: THE golden pattern — preprocessing INSIDE the pipeline ----
    Xs, ys = classification(n=400, d=8, classes=2, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.25,
                                              stratify=ys, random_state=0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),                 # fit on train folds only
        ("clf", LogisticRegression(max_iter=500)),
    ])
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5)
    pipe.fit(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)
    assert scores.mean() > 0.5 and 0.5 < test_acc <= 1.0

    print(f"PASS m01 tooling | cv_acc={scores.mean():.3f}±{scores.std():.3f} "
          f"test_acc={test_acc:.3f} top_city={top_cities.index[0]}")


if __name__ == "__main__":
    main()
