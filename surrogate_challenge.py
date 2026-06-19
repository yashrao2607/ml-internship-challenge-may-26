"""
Surrogate Model Challenge
=========================

In production we serve a heavy `RandomForestRegressor` (the "teacher") trained
on the scikit-learn Diabetes dataset. To cut latency we want to replace it at
serving time with a lightweight, cross-validated XGBoost **surrogate** that
behaves as much like the forest as possible.

What we care about is **fidelity**: how closely the surrogate's outputs track
the teacher's outputs on held-out data. (This is distinct from how well either
model predicts the ground truth.)

This script trains the teacher, fits and tunes the surrogate via GridSearchCV,
and reports fidelity on the test set.

The problem
-----------
The pipeline runs to completion, but the surrogate's fidelity to the forest
comes out well below where it should be. Get the fidelity above the PASS
threshold the script prints.

Rules:
  - Do NOT change the model types, the dataset, the train/test split, the
    param grid, or PASS_THRESHOLD.
  - Keep RANDOM_STATE = 42.
  - A change that *looks* like an improvement can easily make fidelity worse,
    and one fix on its own may not be enough. Be ready to explain why fidelity
    moves the way it does.

Run:  python surrogate_challenge.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42
PASS_THRESHOLD = 0.95

# Grid search over the surrogate can be slow on the full training set, so we
# tune on a representative sample of the rows for speed.
TUNING_SAMPLE_FRAC = 0.5


def load_frame():
    """Load the dataset as a feature frame plus a target series."""
    data = load_diabetes()
    feature_names = list(data.feature_names)
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, feature_names


def fit_teacher(X_train, y_train):
    """Train the black-box forest we later want to approximate."""
    teacher = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    teacher.fit(X_train.values, y_train.values)
    return teacher


def fit_surrogate(train_df, feature_names, teacher):
    """
    Fit and tune the XGBoost surrogate that stands in for the forest.

    Features are standardized before boosting, and hyper-parameters are chosen
    by 5-fold cross-validation.
    """
    # Sample the training rows to keep the grid search fast.
    sample = train_df.sample(frac=TUNING_SAMPLE_FRAC, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(sample[feature_names])
    X = scaler.transform(sample[feature_names])

    # Note for the model card: which inputs the forest leans on most.
    ranked = np.array(feature_names)[np.argsort(teacher.feature_importances_)]
    print(f"Top forest drivers: {[str(f) for f in ranked[-3:][::-1]]}")

    # Regression target for the surrogate.
    target = teacher.predict(sample[feature_names].values)

    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.05, 0.1],
    }
    base = XGBRegressor(
        objective="reg:squarederror",
        # Regularization so individual trees don't chase noise.
        min_child_weight=12,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search = GridSearchCV(base, param_grid, scoring="r2", cv=5, n_jobs=-1)
    search.fit(X, target)

    print(f"Best CV score (R^2): {search.best_score_:.4f}")
    print(f"Best params       : {search.best_params_}")

    # Fit the scaler and the best estimator on the full training set
    scaler = StandardScaler().fit(train_df[feature_names])
    X_full = scaler.transform(train_df[feature_names])
    target_full = teacher.predict(train_df[feature_names].values)

    surrogate = search.best_estimator_
    surrogate.fit(X_full, target_full)

    return surrogate, scaler


def evaluate(teacher, surrogate, scaler, X_test, y_test, feature_names):
    """Report surrogate fidelity to the teacher on the held-out test set."""
    teacher_pred = teacher.predict(X_test.values)
    surrogate_pred = surrogate.predict(scaler.transform(X_test[feature_names]))

    fidelity_r2 = r2_score(teacher_pred, surrogate_pred)
    fidelity_rmse = mean_squared_error(teacher_pred, surrogate_pred) ** 0.5
    teacher_r2 = r2_score(y_test.values, teacher_pred)
    surrogate_r2 = r2_score(y_test.values, surrogate_pred)

    print("\n=== Results (test set) ===")
    print(f"Teacher (RF)  R^2 vs ground truth : {teacher_r2:.4f}")
    print(f"Surrogate     R^2 vs ground truth : {surrogate_r2:.4f}")
    print(f"Surrogate FIDELITY  R^2 vs teacher : {fidelity_r2:.4f}")
    print(f"Surrogate FIDELITY RMSE vs teacher : {fidelity_rmse:.4f}")

    print("\n=== Verdict ===")
    if fidelity_r2 > PASS_THRESHOLD:
        print(f"PASS  fidelity {fidelity_r2:.4f} > {PASS_THRESHOLD} -- "
              "surrogate faithfully reproduces the forest.")
    else:
        print(f"FAIL  fidelity {fidelity_r2:.4f} <= {PASS_THRESHOLD} -- "
              "the surrogate is not tracking the forest closely enough.")


def main():
    X, y, feature_names = load_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    teacher = fit_teacher(X_train, y_train)

    train_df = X_train.copy()
    train_df["target"] = y_train.values

    surrogate, scaler = fit_surrogate(train_df, feature_names, teacher)
    evaluate(teacher, surrogate, scaler, X_test, y_test, feature_names)


if __name__ == "__main__":
    main()
