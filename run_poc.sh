#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_DIR="$ROOT_DIR/poc_integration"
HARDHAT_DIR="$ROOT_DIR/../blindference/fhenix_inference"
TS_SCRIPT="$POC_DIR/1_encrypt_source.ts"
PY_SCRIPT="$POC_DIR/2_train_ppml.py"
MODEL_EXPORT="$POC_DIR/model_export.json"

echo "[1/2] Encrypting toy dataset with CoFHE Hardhat client..."
(
  cd "$HARDHAT_DIR"
  npx hardhat run "$TS_SCRIPT"
)

echo "[2/2] Validating package and exporting mock PPML model..."
python3 "$PY_SCRIPT"

if [[ -f "$MODEL_EXPORT" ]]; then
  echo "PoC completed successfully: $MODEL_EXPORT generated."
else
  echo "PoC failed: model_export.json was not generated." >&2
  exit 1
fi
