#!/usr/bin/env bash
# Downloads the Kaggle mlg-ulb/creditcardfraud dataset into data/.
#
# Prereqs:
#   1. Free Kaggle account.
#   2. API token: Kaggle account settings -> "Create New Token" -> downloads kaggle.json.
#   3. Place it at ~/.kaggle/kaggle.json (chmod 600), or export
#      KAGGLE_USERNAME / KAGGLE_KEY env vars instead.
#
# Usage: ./scripts/download_data.sh

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip

echo "Done. Dataset at data/creditcard.csv"
