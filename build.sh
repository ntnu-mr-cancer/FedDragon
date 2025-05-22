#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# Default model name
DEFAULT_MODEL="joeranbosma/dragon-bert-base-mixed-domain"

# Help message
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [MODEL_NAME]"
    echo
    echo "Builds the Docker image with the specified model name."
    echo "If no model is provided, defaults to: $DEFAULT_MODEL"
    echo
    echo "Examples:"
    echo "  $0 joeranbosma/dragon-roberta-large-domain-specific"
    echo "  $0                         # uses default model"
    exit 0
fi

# Use the provided model name, or fallback to default
MODEL_NAME="${1:-$DEFAULT_MODEL}"
echo "Building docker image with model: $MODEL_NAME"

# Build the Docker image
docker buildx build --build-arg MODEL_NAME="$MODEL_NAME" -t skarrea/feddragonsub:latest --platform=linux/amd64 "$SCRIPTPATH"