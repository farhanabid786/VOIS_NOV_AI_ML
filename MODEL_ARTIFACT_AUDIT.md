# Model artifact audit

- Uploaded dataset: `data_cardiovascular_risk.csv` — 3,390 rows, 17 columns.
- Uploaded notebook target: `TenYearCHD`.
- Notebook final comparison/conclusion identifies Random Forest as the selected model and reports 90.36% in its summary table.
- The individually tuned Random Forest output in notebook cell 155 reports 89.84% accuracy.
- The uploaded `Cardiovascular_Risk_Prediction_Classification.pkl` deserializes as `xgboost.sklearn.XGBClassifier`, with the notebook's XGBoost parameters. It is not the selected Random Forest artifact.
- This deployment now loads the uploaded XGBoost `.pkl` directly from `backend/models/Cardiovascular_Risk_Prediction_Classification.pkl`.
- The API recreates the notebook preprocessing before each prediction: median/mode imputation, sex encoding, log transforms, pulse-pressure engineering and StandardScaler.
- The new holdout evaluation is leakage-aware and differs from the notebook's reported result. See `backend/models/evaluation_metrics.json`.
- SMOTE is fitted only on the training fold by an imbalanced-learn pipeline. Imputation, notebook-aligned log transforms, feature engineering, scaling and classification are serialized together.
- The serialized pipeline is refitted on all supplied rows only after holdout evaluation, for serving predictions.
