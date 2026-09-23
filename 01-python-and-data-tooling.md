# 01 — Python & Data Tooling for AI

> Back to [index](README.md) · Prev: [00 Math](00-mathematical-foundations.md) · Next: [02 ML Foundations](02-machine-learning-foundations.md)

Python is the default AI language. This module covers the working toolkit — not a Python course, but what you must be fluent in to do everything that follows.

## 1. Python essentials you actually use

```python
# Data structures you'll live in
xs = [1, 2, 3]            # lists
d = {"a": 1}              # dicts
s = {1, 2, 3}             # sets — fast membership
t = (1, 2)                # tuples — immutable records

# Comprehensions — the lingua franca of data munging
squares = [x*x for x in xs if x > 1]

# Functions, defaults, *args/**kwargs
def f(x, *args, scale=1.0, **kwargs): ...

# Classes (frameworks are class-based: torch.nn.Module, sklearn BaseEstimator)
class Model:
    def __init__(self, lr): self.lr = lr
    def fit(self, X, y): ...

# Key stdlib: pathlib, json, dataclasses, typing, itertools, collections, functools, re, random
```

**Fluency bar:** type hints, context managers (`with`), generators (`yield`), lambda/map, virtual environments (`venv`/`uv`), `pip`/`pyproject.toml`.

## 2. NumPy — matrices on your CPU

```python
import numpy as np
a = np.array([[1., 2.], [3., 4.]])   # shape (2,2), dtype float64
a @ b        # matmul            a.T  # transpose
a.sum(axis=0)                     # column sums
np.random.randn(3, 4)             # standard normal init
np.linalg.norm(a), np.linalg.inv(a)
mask = a > 2                      # boolean indexing → a[mask]
b = np.broadcast_to(a, (4, 2, 2)) # broadcasting rules
```

**Master:** vectorization (no Python loops over data), axis semantics, broadcasting, views vs copies, `reshape`/`transpose`, random API, float32 vs float64 memory tradeoffs.

## 3. pandas — tables

```python
import pandas as pd
df = pd.read_csv("data.csv")
df.head(); df.info(); df.describe()
df["age"] > 30                    # boolean filter
df[df.city == "Paris"]
df.groupby("city")["sales"].mean()
df["log_sales"] = np.log1p(df["sales"])
df = df.merge(other, on="user_id", how="left")
df.isna().sum()                   # missing values
df["ts"] = pd.to_datetime(df["ts"])
df.to_parquet("clean.parquet")    # fast columnar format
```

**Master:** selection (`loc`/`iloc`), groupby-aggregation, joins, missing-data strategy, dtypes/categories for memory.

## 4. Visualization — see the data, see the training

```python
import matplotlib.pyplot as plt
df.age.hist(bins=40); plt.xlabel("age"); plt.show()
plt.scatter(x, y, c=labels)                 # relationships
# Training curves (you will plot these every day):
plt.plot(train_loss, label="train"); plt.plot(val_loss, label="val"); plt.legend()
```

Diagnosis toolkit: histogram (distributions), scatter (relationships), boxplot (outliers), correlation heatmap, confusion matrix heatmap, loss curves.

## 5. Jupyter & environment hygiene

- Jupyter/VS Code notebooks for exploration; **scripts + tests for real work.**
- Reproduce order: seed RNGs (`random`, `np.random`, framework seeds), record package versions (`pip freeze`), pin data.
- Env tools: `venv` or `uv`; never install into system Python.

## 6. scikit-learn — the classic ML workflow

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", RandomForestClassifier(random_state=42))])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
print(cross_val_score(pipe, X_train, y_train, cv=5))
```

**The golden rule this encodes:** preprocessing goes *inside* the pipeline, fit only on training folds. Fitting a scaler on all data before splitting is leakage (see [05](05-model-evaluation-and-tuning.md)).

## 7. The wider toolchain (map for later modules)

| Need | Tool |
|---|---|
| Deep learning | PyTorch (default research), TensorFlow/Keras (production legacy) |
| LLM APIs | `openai`, `anthropic`, provider SDKs |
| Vector search | FAISS, chromadb, pgvector, Qdrant, Weaviate |
| Experiment tracking | MLflow, W&B, TensorBoard |
| Orchestration | Airflow, Prefect, Dagster |

## Mastery Checklist

- [ ] Can vectorize a NumPy operation instead of writing a loop
- [ ] Can do groupby, merge, and missing-value handling without looking up docs
- [ ] Builds sklearn Pipelines with cross-validation and never leaks preprocessing
- [ ] Plots and interprets training/validation loss curves
- [ ] Has a reproducible environment (seeds + pinned dependencies)
