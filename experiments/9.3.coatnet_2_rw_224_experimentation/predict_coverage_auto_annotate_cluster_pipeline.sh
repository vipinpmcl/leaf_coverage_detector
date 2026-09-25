#!/bin/bash

set -e

EXP_ID="exp11_9"
CONFIG="runs/${EXP_ID}/config.yaml"
CHECKPOINT="runs/${EXP_ID}/best.pt"

BASE_DATA="../../../../datasets/20CropData/20Crops200Samples_Watermarked/20Crops200Samples_Watermarked"


for CROP in "$@"; do

    IMAGE_DIR="$BASE_DATA/$CROP"
    OUTPUT_DIR="runs/${EXP_ID}/${CROP,,}"

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

    # 3. Refine annotations based on leaf coverage
    python tools/auto_annotation_based_on_leaf_coverage_v5/scripts/refine_batch.py --config tools/auto_annotation_based_on_leaf_coverage_v5/configs/default.yaml --input-dir runs/${EXP_ID}/${CROP,,}/
    
    
    # 4. Cluster based on leaf coverage
    python tools/clusters_data_based_on_coverage_v2.py \
        --csv "$OUTPUT_DIR/leaf_coverage.csv" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUTPUT_DIR/clusters_output" \
        --n-clusters 5 \
        --copy

    echo ""
    echo "Completed: $CROP"

done

echo ""
echo "========================================"
echo "ALL CROPS COMPLETED"
echo "========================================"