#!/bin/bash

set -e

EXP_ID="exp14_7_pigeonpea_soybean_rep_images_added"
CONFIG="runs/${EXP_ID}/config.yaml"
CHECKPOINT="runs/${EXP_ID}/best.pt"

# BASE_DATA="../../../../datasets/20CropData/20Crops200Samples_Watermarked/20Crops200Samples_Watermarked"
# BASE_DATA="../../../../datasets/disease_august/Potato/Potato___healthy"
BASE_DATA="data/disease_aug_data_temp/cotton_mix_dinov2_hier/"


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

    # 0. Rename images with hash to avoid duplicates and invalid characters in filenames
    python tools/rename_images_with_hash.py "$IMAGE_DIR"
    
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

    #3. Refine annotations based on leaf coverage
    python tools/auto_annotation_based_on_leaf_coverage_v5/scripts/refine_batch.py --config tools/auto_annotation_based_on_leaf_coverage_v5/configs/default.yaml --input-dir runs/${EXP_ID}/${CROP,,}/
    
    
    # 4. Cluster based on leaf coverage
    python tools/clusters_data_based_on_coverage_v2.py \
        --csv "$OUTPUT_DIR/leaf_coverage.csv" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$OUTPUT_DIR/clusters_output_sam2_0_1_0p6" \
        --n-clusters 2 \
        --copy

    echo ""
    echo "Completed: $CROP"

done

echo ""
echo "========================================"
echo "ALL CROPS COMPLETED"
echo "========================================"