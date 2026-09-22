# CoAtNet2 Mask -> SAM2 Refinement -> LabelMe

This project refines existing CoAtNet2 binary leaf masks with SAM2. It does NOT search for additional leaves.

## Input

Recursively discovers sample directories containing:
- original.jpg
- mask.png

Directories named `clusters_output` are skipped.

Example:

predictions/
  cauliflower/
    sample_001/
      original.jpg
      mask.png

## Output

For each sample, writes:

sample_001/
  sample_001.json
  sam2_refined_mask.png
  sam2_overlay.jpg
  sam2_refinement.json

The JSON is LabelMe 5.11.3 compatible and embeds the original image as Base64.

## Run

python scripts/refine_batch.py \
  --config configs/default.yaml \
  --input-dir ../../runs/exp9/cauliflower/test/ \
  --sam-checkpoint /d/Vipin/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overwrite

The SAM2 Python package/repository must already be installed/importable in the environment.
