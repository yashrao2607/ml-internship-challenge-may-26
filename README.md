# Surrogate Model Challenge Solutions

A lightweight XGBoost surrogate model was successfully trained to approximate the outputs of a heavy `RandomForestRegressor` (the teacher model) on the scikit-learn Diabetes dataset.

## Final Results (Test Set)

- **Teacher (RF) $R^2$ vs Ground Truth**: `0.4523`
- **Surrogate $R^2$ vs Ground Truth**: `0.4758`
- **Surrogate FIDELITY $R^2$ vs Teacher**: `0.9847` (PASS, threshold is `0.95`)
- **Surrogate FIDELITY RMSE vs Teacher**: `6.3352`

---

## Technical Explanation & Fixes

To achieve a fidelity score above the required **95%** threshold, the following issues were resolved in [surrogate_challenge.py](file:///d:/omli%20-ai/surrogate_challenge.py):

### 1. Correcting the Training Target
* **Problem**: The original implementation trained the XGBoost surrogate using the ground-truth targets (`y_train`).
* **Fix**: A surrogate model's purpose is to replicate the behavior and predictions of the teacher model (Random Forest), not the original target. The training target for the surrogate was updated to use the teacher's predictions (`teacher.predict(X_train)`).
* **Impact**: This change alone improved test-set fidelity from `0.7827` to `0.9286`.

### 2. Retraining on the Full Dataset
* **Problem**: The grid search was conducted on a 50% subset (`TUNING_SAMPLE_FRAC = 0.5`) to optimize search speeds. However, the final model was never retrained on the full dataset, meaning half of the training data was wasted.
* **Fix**: After finding the optimal hyperparameters using `GridSearchCV` on the subset, the `StandardScaler` was refit on the full training set, and the best estimator was retrained on the complete set of teacher predictions.
* **Impact**: Utilizing 100% of the training data boosted the final test-set fidelity from `0.9286` to `0.9847` (well above the `0.95` pass criteria).
