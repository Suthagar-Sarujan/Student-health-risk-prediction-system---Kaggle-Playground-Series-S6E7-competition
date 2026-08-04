# Student Health Risk Prediction — CIS6005 Computational Intelligence

Predicts student health status (`at-risk` / `fit` / `unhealthy`) from
lifestyle and physiological features, using the Kaggle Playground
Series S6E7 "Student Health Risk" dataset (690,088 training records).

Six Computational Intelligence techniques are trained and compared
under identical preprocessing: Logistic Regression (baseline), Random
Forest, XGBoost, LightGBM, CatBoost, and a Neural Network (MLP).
LightGBM is deployed via a Streamlit web application.

## Project structure

```
student-health-risk-project/
├── data/               # train.csv / test.csv (download from Kaggle — see data/README.md)
├── notebooks/          # optional exploratory notebook
├── src/                # all training, evaluation, and preprocessing code
├── models/             # generated: fitted pipeline + all 6 trained models
├── app/                # Streamlit deployment app
├── figures/            # generated: EDA, evaluation, and SHAP plots
├── submission_lightgbm.csv   # generated: final Kaggle submission
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the Kaggle competition page
and place them in `data/` (see `data/README.md` for details).

## Running the pipeline

All scripts are run from the project root, with `src/` on the path
(or `cd src` first). Order matters: the first training script fits
and saves the shared preprocessing pipeline; every subsequent script
reuses it.

```bash
cd src

# 1. Exploratory data analysis — generates Figures 1-5
python eda.py

# 2. Train all six models (order doesn't matter after the first run,
#    but running logistic regression first is a sensible default —
#    it fits and saves the shared preprocessor)
python train_logistic_regression.py
python train_random_forest.py
python train_xgboost.py
python train_lightgbm.py
python train_catboost.py
python train_neural_network.py

# 3. Compare all six models — generates Figures 7-8 + comparison table
python evaluate_models.py

# 4. SHAP / feature importance for the deployed LightGBM model — Figures 9-10
python explainability.py

# 5. Generate the Kaggle leaderboard submission
python generate_submission.py
```

## Running the app

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Fill in the 13 lifestyle/
physiological inputs and click **Predict health risk** to see the
predicted class and the full 3-class probability breakdown.

## Notes

- The preprocessing pipeline (median imputation + StandardScaler for
  numeric features; "missing"-category imputation + OneHotEncoder for
  categorical features) is defined once in `src/preprocessing.py` and
  imported by every training script and by the Streamlit app — this is
  what guarantees no train/serve skew.
- All class-imbalance handling (`class_weight='balanced'` /
  `sample_weight` / `auto_class_weights='Balanced'`) is applied
  per-model in each `train_*.py` script, except the MLP, which
  scikit-learn does not support per-sample weighting for (see the
  report's Section F discussion of this limitation).
- `model_metrics.json` is appended to by each training script and is
  the single source of truth for the Section E comparison table —
  regenerate it by re-running the relevant `train_*.py` script.
