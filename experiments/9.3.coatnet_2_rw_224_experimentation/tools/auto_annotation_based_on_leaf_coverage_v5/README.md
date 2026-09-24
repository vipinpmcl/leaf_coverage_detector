# CoAtNet2 -> SAM2 -> LabelMe v4

Coverage-aware auto-annotation pipeline for existing CoAtNet2 predictions.

For every sample directory, the script reads `leaf_coverage.txt` and routes the image using configurable thresholds:

```text
leaf coverage < LOW_THRESHOLD
        -> one LabelMe POINT with configurable negative label

LOW_THRESHOLD <= leaf coverage <= HIGH_THRESHOLD
        -> IGNORE auto-annotation, COPY IMAGE for manual LabelMe review

leaf coverage > HIGH_THRESHOLD and <= 100%
        -> CoAtNet2 mask -> SAM2 refinement -> LabelMe POLYGON(s)
```

The boundary convention above deliberately removes gaps at exactly the configured thresholds. Thus, if `low=5` and `high=20`:

- `4.999%` -> negative point
- `5.0%` -> ignored
- `20.0%` -> ignored
- `20.001%` -> SAM2 polygon
- `100%` -> SAM2 polygon

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
  negative_label: negative
  negative_description: "Leaf coverage below low threshold"
  point_source: mask_centroid
  empty_mask_fallback: error
```

### Point placement

`point_source: mask_centroid` places the negative point at the centroid of all nonzero pixels in the existing CoAtNet2 mask. This is independent of the normal SAM2 component-area threshold.

If the mask is empty, the default is to raise an error. To use the image center instead:

```yaml
empty_mask_fallback: image_center
```

Alternatively:

```yaml
point_source: image_center
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
    └── <sample-id>.json
```

Low-coverage sample JSON contains a standard LabelMe point shape:

```json
{
  "label": "negative",
  "points": [[x, y]],
  "group_id": null,
  "description": "Leaf coverage below low threshold",
  "shape_type": "point",
  "flags": {},
  "mask": null
}
```

High-coverage samples contain the existing SAM2-refined polygon annotations.

`imageData` is embedded as Base64 by default and is validated before writing, to avoid the previous LabelMe `NoneType` image-data issue.

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
- `annotation_mode` (`negative_point`, `ignore`, `sam2_polygon`, `error`)
- ignored samples have `status=ignored_manual_review` because their images are copied for manual annotation
- `status`
- `input_components`
- `output_polygons`
- `output_points`
- `error`
