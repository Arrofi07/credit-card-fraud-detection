# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project
End-to-end credit card fraud detection using the Kaggle `mlg-ulb/creditcardfraud`
dataset. See PRD.md for full scope, goals, and phased roadmap.

## Stack (free/no-cost only — do not introduce paid services)
- Python 3.11+, managed via venv
- pandas / numpy / scikit-learn / imbalanced-learn / xgboost (or lightgbm)
- FastAPI + uvicorn for the prediction API
- Streamlit for the demo UI
- pytest for tests
- Docker + docker-compose for local containerization
- MLflow (local file store) for experiment tracking, if used

## Working agreements
- This is a learning/portfolio project — prefer clear, well-structured code
  over maximal sophistication. A simple, well-evaluated model beats a complex
  under-evaluated one.
- Never commit the raw dataset (`data/`), trained model artifacts, or `mlruns/`.
  These are gitignored; keep it that way.
- This dataset is severely imbalanced (~0.17% fraud). Never report or optimize
  plain accuracy as a success metric. Always report precision, recall, F1, and
  PR-AUC for the fraud class, plus a confusion matrix.
- Keep EDA in `notebooks/`, reusable pipeline code in `src/`. Notebooks should
  import from `src/`, not duplicate logic.
- Deployment target for now is local only (Docker Compose running the API +
  Streamlit app). Do not add cloud deployment config unless asked.
- Follow the phased roadmap in PRD.md; don't jump ahead to later phases
  (e.g. CI, cloud hosting, monitoring) without being asked.

## Running things (once scaffolded)
- API: `uvicorn src.api.main:app --reload`
- UI: `streamlit run src/app/demo.py`
- Tests: `pytest`
