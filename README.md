# Credit Card Fraud Detection

End-to-end fraud detection on the Kaggle [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset — data → model → API → demo UI, using only free/no-cost tooling.

See [PRD.md](PRD.md) for scope and roadmap, and [CLAUDE.md](CLAUDE.md) for
working agreements in this repo.

Status: Phase 6 (Containerization) done — all core phases complete. See
Future Work in PRD.md for what's intentionally left out.

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

## Run the API

```bash
uvicorn src.api.main:app --reload
```

Swagger docs at http://127.0.0.1:8000/docs. `POST /predict` expects the 30
dataset fields (`Time`, `V1`-`V28`, `Amount`) and returns a fraud
probability, a boolean decision, and the threshold applied.

## Run the demo UI

With the API running (see above), in another terminal:

```bash
streamlit run src/app/demo.py
```

Sample a legit or fraudulent transaction from the real dataset and send it
to the API to see the prediction. Set `API_URL` if the API isn't on the
default `http://127.0.0.1:8000`.

## Run everything with Docker

Requires `models/fraud_model.joblib` and `data/creditcard.csv` to already
exist on the host (see Train/Get the data above) — they're mounted into the
containers, not baked into the image.

```bash
docker compose up --build
```

API on http://localhost:8000, UI on http://localhost:8501 (reaching the API
at `http://api:8000` over the compose network).

## Tests

```bash
pytest
```
