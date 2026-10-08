# PRD: Credit Card Fraud Detection

## 1. Problem & Goal
Build an end-to-end, portfolio-quality fraud detection system on the Kaggle
`mlg-ulb/creditcardfraud` dataset: from raw data to a model that flags
fraudulent transactions, exposed through a REST API with an interactive demo
UI, running locally via free/no-cost tooling.

## 2. Dataset
- Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions, 2 days, European cardholders, 492 fraud (~0.172%).
- Features: `Time`, `Amount`, `V1`–`V28` (PCA-anonymized), `Class` (label).
- No missing values. Not bundled in repo — downloaded via Kaggle API into
  `data/` (gitignored).

## 3. Non-goals (out of scope for now)
- Paid infrastructure of any kind.
- Real-time streaming ingestion (dataset is static/historical).
- Cloud deployment (deferred to Future Work).
- Multi-model ensembling / AutoML beyond comparing a couple of models.

## 4. Success Criteria
- A trained model with PR-AUC and recall-at-fixed-precision reported and
  justified (not accuracy).
- A documented, reproducible pipeline (clear steps from raw CSV to model
  artifact).
- A working FastAPI endpoint that returns a fraud probability + decision for
  a submitted transaction.
- A Streamlit demo where a user can submit a transaction (or sample one from
  the test set) and see the prediction.
- Everything runnable locally with a single documented command sequence
  (or `docker-compose up`).

## 5. Users
Primarily the author (portfolio piece, learning project) and anyone reviewing
the repo (recruiters, collaborators) who wants to see the model working
end-to-end without needing paid services or cloud accounts.

## 6. Phased Roadmap
1. **Bootstrap** (done): CLAUDE.md, PRD.md, repo scaffold, git init.
2. **Data & EDA** (done): Kaggle download script, EDA notebook (class
   balance, Amount/Time behavior, V1–V28 separation). See
   `notebooks/01_eda.ipynb`.
3. **Modeling** (done): compared class-weighted Logistic Regression, SMOTE +
   Logistic Regression, and class-weighted XGBoost in
   `notebooks/02_modeling.ipynb`. XGBoost won clearly (PR-AUC 0.853 vs.
   ~0.72 for both Logistic Regression variants); SMOTE gave no measurable
   edge over class weighting, so class weighting (`scale_pos_weight`) is the
   shipped imbalance strategy (see `src/train.py`). At the best-F1
   threshold (0.940): precision 0.909, recall 0.816, F1 0.860 on the held-out
   test set (8 false positives, 18 false negatives out of 56,962 rows).
   `scripts/train_model.py` reproduces this end-to-end and saves
   `models/fraud_model.joblib` + `reports/metrics.json`.
4. **API**: FastAPI service wrapping the saved model, `/predict` endpoint,
   input validation, Swagger docs.
5. **Demo UI**: Streamlit app that calls the API, lets a user try sample or
   custom transactions, shows the prediction and probability.
6. **Containerization**: Dockerfile(s) + docker-compose for API + UI,
   running fully locally.
7. **Future work (not started yet)**: experiment tracking with MLflow, test
   suite expansion, CI via GitHub Actions, free cloud hosting (Hugging Face
   Spaces for the UI, Render/Railway free tier for the API), model monitoring
   for drift.

## 7. Risks / Open Questions
- SMOTE vs. class-weighting choice affects precision/recall tradeoff —
  decide and document in Phase 3, not before.
- PCA-anonymized features limit interpretability (e.g. SHAP will explain
  "V14 contributed X" without real-world meaning) — acceptable for this
  dataset, call out in README.
