# CoAtNet2 -> SAM2 -> binary masks

Coverage-aware auto-annotation pipeline for existing CoAtNet2 predictions.

For every sample directory, the script reads `leaf_coverage.txt` and routes the image using configurable thresholds:

```text
leaf coverage < LOW_THRESHOLD
        -> empty negative MASK PNG

LOW_THRESHOLD <= leaf coverage <= HIGH_THRESHOLD
        -> IGNORE auto-annotation, COPY IMAGE for manual LabelMe review

leaf coverage > HIGH_THRESHOLD and <= 100%
        -> CoAtNet2 mask -> SAM2 refinement -> binary MASK PNG
```

The boundary convention above deliberately removes gaps at exactly the configured thresholds. Thus, if `low=5` and `high=20`:

- `4.999%` -> empty negative mask
- `5.0%` -> ignored
- `20.0%` -> ignored
- `20.001%` -> SAM2 mask PNG
- `100%` -> SAM2 mask PNG

## Input

Each sample directory is expected to contain:

```text
<sample-id>/
├── original.jpg
├── mask.png
└── leaf_coverage.txt
```

The coverage file can contain, for example:

```text
Leaf Coverage: 3.4815%
```

Directories named `clusters_output` are skipped recursively.

## Configuration

Edit `configs/default.yaml`:

```yaml
coverage:
  enabled: true
  low_threshold_percent: 5.0
  high_threshold_percent: 20.0
```

## Important: ignored samples

Samples in the middle coverage range are intentionally not auto-annotated. Their **image is still copied** into the output directory so you can open it in LabelMe and annotate it manually. No JSON is generated automatically for these samples.

The copied image uses the same per-sample layout as the other outputs:

```text
auto_annotation_output/
└── <sample-id>/
    └── <sample-id>.jpg
```

The summary CSV records these samples with `status=ignored_manual_review`. With `--overwrite`, the copied image is refreshed.

## Output

Annotated samples use the same v3 layout:

```text
auto_annotation_output/
└── <sample-id>/
    ├── <sample-id>.jpg
    └── <sample-id>_mask.png      # low-coverage empty or high-coverage SAM2 mask
```

Low-coverage masks are all black (`0`) and have the same dimensions as the source mask. High-coverage masks use white (`255`) for foreground and black (`0`) for background.

## Run

```bash
python scripts/refine_batch.py \
  --config configs/default.yaml \
  --input-dir ../../runs/exp9/cauliflower/test/ \
  --output-dir auto_annotation_output \
  --sam-checkpoint /d/Vipin/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam-config /d/Vipin/github_repos/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

SAM2 is loaded **only if at least one selected sample is in the high-coverage range**. Point and ignored samples do not invoke SAM2.

## Summary

The batch writes:

```text
auto_annotation_output/sam2_labelme_summary.csv
```

Important columns:

- `leaf_coverage_percent`
- `annotation_mode` (`negative_mask`, `ignore`, `sam2_mask`, `error`)
- ignored samples have `status=ignored_manual_review` because their images are copied for manual annotation
- `status`
- `input_components`
- `output_masks`
- `error`
