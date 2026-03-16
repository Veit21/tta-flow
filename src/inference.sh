#!/bin/bash
####################################################################################
#
#   Runs inference using a trained model checkpoint and Hydra configuration.
#
####################################################################################

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the user provided the required experiment_path argument
if [ "$#" -lt 1 ]; then
    echo "Error: Missing experiment_path."
    echo "Usage: ./src/inference.sh <path_to_experiment_dir> [data_config]"
    echo "Example: ./src/inference.sh runs/2026-03-05/10-00-00 retouch_cirrus"
    exit 1
fi

EXPERIMENT_PATH=$1

# Default to retouch_cirrus
DATA_CONFIG=${2:-retouch_cirrus} 

echo "Starting inference script: $SCRIPT_DIR/inference.sh"
echo "Experiment Path: $EXPERIMENT_PATH"
echo "Dataset Config: $DATA_CONFIG"

# Run inference with Hydra overrides
python "$SCRIPT_DIR/test.py" \
    +experiment_path="$EXPERIMENT_PATH" \
    data="$DATA_CONFIG" \
    hydra.run.dir="${EXPERIMENT_PATH}/inference" \
    "${@:3}"

echo "Inference finished successfully."