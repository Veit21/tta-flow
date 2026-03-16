#!/bin/bash
####################################################################################
#
#   Trains the Flow Matching model using Hydra configuration.
#
####################################################################################

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset configuration - defaults to retouch_spectralis
DATA_CONFIG=${1:-retouch_spectralis}

# Experiment comment/name - defaults to 'baseline_run'
COMMENT=${2:-baseline_run}

echo "Starting training script: $SCRIPT_DIR/train.sh"
echo "Dataset Config: $DATA_CONFIG"
echo "Experiment Comment: $COMMENT"

# Run training with Hydra overrides
python "$SCRIPT_DIR/train.py" \
    data="$DATA_CONFIG" \
    train_parameters.comment="$COMMENT" \
    "${@:3}"

echo "Training finished successfully."