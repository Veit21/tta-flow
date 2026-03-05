#!/bin/bash
####################################################################################
#
#   Flow Matching Training Script
#   Trains the Flow Matching model using Hydra configuration.
#
####################################################################################

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting training script: $SCRIPT_DIR/train.sh"

# Run training with Hydra config from preferences/config.yaml
python "$SCRIPT_DIR/train.py" train_parameters.comment=spectralis_224

echo "Training finished successfully."