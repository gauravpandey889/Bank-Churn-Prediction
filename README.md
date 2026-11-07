# Bank Customer Churn Intelligence

## Quick start

Python 3.10+ recommended. In this directory:

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

The trained model and precomputed outputs are included, so the dashboard can be started after installing requirements. The application was prepared and syntax-checked but not browser-tested in this environment.

## Contents
- `European_Bank.csv`: supplied dataset (10,000 records)
- `train.py`: reproducible preprocessing, engineered features, training and evaluation
- `train_utils.py`: same feature engineering for app inference
- `app.py`: Streamlit dashboard with overview, risk calculator, what-if simulator, model explainability, evaluation
- `models/`: trained model and metadata
- `outputs/`: risk scores, comparisons, importance, figures
- `Research_Report.pdf` and `Executive_Summary.pdf`: submission documents

## Methodological notes
- The target is `Exited`; `Year` is constant and excluded. `CustomerId` and `Surname` are excluded.
- 80/20 stratified train/test split. Binary threshold chosen on a separate training validation split to avoid test-set tuning.
- Model selected by test ROC-AUC, so the reported best-model performance is subject to model-selection optimism; a further untouched holdout or external cohort is preferable for production.
- Risk bands are illustrative; probabilities are not explicitly calibrated. The what-if tool is associational, not causal.
- Gender and geography are used as features in this educational project; assess fairness and legal appropriateness before production use.
- SHAP and partial dependence plots are optional extensions and are not included. Permutation feature importance is included.
- No live bank data or live deployment is claimed.
