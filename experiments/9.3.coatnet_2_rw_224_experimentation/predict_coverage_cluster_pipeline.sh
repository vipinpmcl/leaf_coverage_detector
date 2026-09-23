#!/bin/bash

set -e

CONFIG="runs/exp11_5/config.yaml"
CHECKPOINT="runs/exp11_5/best.pt"

BASE_DATA="../../../../datasets/20CropData/20Crops200Samples_Watermarked/20Crops200Samples_Watermarked"

if [ "$#" -eq 0 ]; then
    echo "Usage:"
    echo "  ./run_exp11_5_crop.sh <crop1> [crop2] [crop3] ..."
    echo ""
    echo "Example:"
    echo "  ./run_exp11_5_crop.sh Potato Cotton Groundnut"
    exit 1
fi

for CROP in "$@"; do

    IMAGE_DIR="$BASE_DATA/$CROP"
    OUTPUT_DIR="runs/exp11_5/${CROP,,}"

    echo ""
    echo "========================================"
    echo "Processing: $CROP"
    echo "Input:     $IMAGE_DIR"
    echo "Output:    $OUTPUT_DIR"
    echo "========================================"

    # Check input directory
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "ERROR: Input directory does not exist:"
        echo "$IMAGE_DIR"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    # 1. Prediction
    python batch_predict.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUTPUT_DIR"

    # 2. Calculate leaf coverage
    python tools/calculate_leaf_coverage.py \
        --input-dir "$OUTPUT_DIR/" \
        --output-dir "$OUTPUT_DIR/"

    # 3. Cluster based on leaf coverage
    python tools/clusters_data_based_on_coverage.py \
        --csv "$OUTPUT_DIR/leaf_coverage.csv" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUTPUT_DIR/clusters_output" \
        --n-clusters 3 \
        --copy

    echo ""
    echo "Completed: $CROP"

done

echo ""
echo "========================================"
echo "ALL CROPS COMPLETED"
echo "========================================"