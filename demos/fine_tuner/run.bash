#! /bin/bash

# Enable job control and set process group
set -m
trap 'kill $(jobs -p)' EXIT INT TERM

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/finetune.yaml"

# Parse command line arguments
usage() {
    echo "Usage: $0 [-c|--config <config_path>] [-n|--num-gpus <num_gpus>]"
    echo "  -c, --config     Path to config file (default: ${DEFAULT_CONFIG})"
    echo "  -n, --num-gpus   Number of GPUs to use (default: 8)"
    exit 1
}

# Default values
CONFIG_PATH="${DEFAULT_CONFIG}"
NUM_GPUS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate config file exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config file not found at ${CONFIG_PATH}"
    exit 1
fi

# Validate num_gpus is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Number of GPUs must be a positive integer"
    exit 1
fi

# Set distributed training environment variables
export MASTER_PORT=29500
export MASTER_ADDR="localhost"
export WORLD_SIZE=$NUM_GPUS
export TF_CPP_MIN_LOG_LEVEL=3
export COMPILE_DIT=1

# Set IS_DISTRIBUTED based on NUM_GPUS
if [ "$NUM_GPUS" -gt 1 ]; then
    export IS_DISTRIBUTED=true
fi

# Load .env file (if it exists)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Starting training with ${NUM_GPUS} GPU(s), mode: ${IS_DISTRIBUTED:+distributed}${IS_DISTRIBUTED:-single_gpu}"
echo "Using config: ${CONFIG_PATH}"

# Launch processes
if [ "$NUM_GPUS" -gt 1 ]; then
    for RANK in $(seq 0 $((NUM_GPUS-1))); do
        env RANK=$RANK CUDA_VISIBLE_DEVICES=$RANK python "${SCRIPT_DIR}/train.py" --config-path "${CONFIG_PATH}" &
    done
else
    python "${SCRIPT_DIR}/train.py" --config-path "${CONFIG_PATH}" &
fi

# Wait for all background processes to complete
wait

# Check if any process failed
if [ $? -ne 0 ]; then
    echo "One or more training processes failed"
    exit 1
fi