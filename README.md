# Credit Card Fraud Detection

End-to-end fraud detection on the Kaggle [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset — data → model → API → demo UI, using only free/no-cost tooling.

See [PRD.md](PRD.md) for scope and roadmap, and [CLAUDE.md](CLAUDE.md) for
working agreements in this repo.

Status: Phase 3 (Modeling) done. API and UI are not yet implemented. See the
phased roadmap in PRD.md.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Requires a free Kaggle account and API token (Kaggle account settings →
"Create New Token" → saves `kaggle.json` → place at `~/.kaggle/kaggle.json`).

```bash
./scripts/download_data.sh
```

This downloads `creditcard.csv` into `data/` (gitignored).

## Explore & model

```bash
jupyter notebook notebooks/01_eda.ipynb      # EDA
jupyter notebook notebooks/02_modeling.ipynb # model comparison
```

## Train the shipped model

```bash
python scripts/train_model.py
```

Trains the class-weighted XGBoost model, saves `models/fraud_model.joblib`
(gitignored) and `reports/metrics.json` (tracked).

## Tests

```bash
pytest
```
