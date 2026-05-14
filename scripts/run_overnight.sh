#!/usr/bin/env bash
# Overnight pipeline:
#   1. Train dictabert and heBERT for 6 epochs each on the calibration_v2 labels.
#   2. Pick the higher macro-F1 winner and save it.
#   3. Run the saved classifier over the full 33K-row corpus.
#   4. Write a human-readable summary to logs/overnight_summary.md.
#
# Run from the repo root:  ./scripts/run_overnight.sh
set -eo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

banner() {
    printf '\n=== [%s] %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

trap 'banner "FAILED at line $LINENO (exit $?)"; exit 1' ERR

banner "Step 1/3: B.4a comparison (dictabert vs heBERT, 6 epochs, save winner)"
python src/finetune_compare.py --epochs 6 --save 2>&1 | tee logs/b4a_full.log

banner "Step 2/3: Classify full corpus with saved winner"
python src/classify_corpus.py 2>&1 | tee logs/full_corpus_inference.log

banner "Step 3/3: Write summary"
python scripts/write_overnight_summary.py 2>&1 | tee logs/overnight_summary.log

banner "DONE"
