from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components import connected_components, remove_small_components  # noqa: E402
from src.coverage import classify_coverage, read_leaf_coverage  # noqa: E402
from src.input_loader import discover_prediction_samples, read_image, read_mask  # noqa: E402
from src.labelme import save_labelme_json  # noqa: E402
from src.polygons import mask_to_polygons  # noqa: E402
from src.visualization import save_mask, save_overlay  # noqa: E402


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value, config_path):
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return Path(config_path).resolve().parent / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create LabelMe annotations from CoAtNet2 masks. "
            "Low coverage becomes a negative point, the middle range is ignored, "
            "and high coverage is refined with SAM2 polygons."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mask_centroid(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def image_center(image_rgb: np.ndarray) -> list[float]:
    height, width = image_rgb.shape[:2]
    return [float((width - 1) / 2.0), float((height - 1) / 2.0)]


def get_negative_point(mask, image_rgb, source, empty_fallback):
    if source == "mask_centroid":
        point = mask_centroid(mask)

        if point is not None:
            return point

        # Empty mask fallback
        if empty_fallback == "image_center":
            return image_center(image_rgb)

        if empty_fallback == "error":
            raise RuntimeError(
                "Cannot create negative point: mask is empty."
            )

        raise ValueError(
            f"Unsupported coverage.empty_mask_fallback: {empty_fallback}"
        )

    if source == "image_center":
        return image_center(image_rgb)

    raise ValueError(
        f"Unsupported coverage.point_source: {source}"
    )


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    coverage_cfg = cfg.get("coverage", {})
    sam_cfg = cfg.get("sam2", {})
    component_cfg = cfg.get("components", {})
    refinement_cfg = cfg.get("refinement", {})
    polygon_cfg = cfg.get("polygon", {})
    visualization_cfg = cfg.get("visualization", {})
    labelme_cfg = cfg.get("labelme", {})
    device = cfg.get("device", "cuda")

    output_dir = Path(
        args.output_dir if args.output_dir is not None else output_cfg.get("directory", "auto_annotation_output")
    )
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    low_threshold = float(coverage_cfg.get("low_threshold_percent", 5.0))
    high_threshold = float(coverage_cfg.get("high_threshold_percent", 20.0))
    if not (0 <= low_threshold < high_threshold <= 100):
        raise ValueError(
            "coverage thresholds must satisfy "
            "0 <= low_threshold_percent < high_threshold_percent <= 100"
        )

    samples = list(
        discover_prediction_samples(
            args.input_dir,
            original_name=input_cfg.get("original_name", "original.jpg"),
            mask_name=input_cfg.get("mask_name", "mask.png"),
            skip_directories=input_cfg.get("skip_directories", ["clusters_output"]),
        )
    )
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input dir    : {Path(args.input_dir).resolve()}")
    print(f"Output dir   : {output_dir}")
    print(f"Coverage     : < {low_threshold}% point-negative; {low_threshold}%..{high_threshold}% ignore; > {high_threshold}% SAM2")
    print(f"Samples      : {len(samples)}")

    if not samples:
        print("No valid samples found.")
        return

    # Parse coverage before constructing SAM2. This avoids loading a large SAM2 model
    # when the selected batch contains only point/ignored samples.
    prepared = []
    for sample in samples:
        sample_id = sample["sample_dir"].name
        coverage_path = sample["sample_dir"] / input_cfg.get("coverage_name", "leaf_coverage.txt")
        try:
            coverage = read_leaf_coverage(coverage_path)
            mode = classify_coverage(coverage, low_threshold, high_threshold)
            prepared.append({**sample, "coverage": coverage, "mode": mode, "coverage_error": ""})
        except Exception as exc:
            prepared.append({**sample, "coverage": None, "mode": "error", "coverage_error": f"{type(exc).__name__}: {exc}"})

    sam_needed = any(item["mode"] == "sam2_polygon" for item in prepared)
    refiner = None

    if sam_needed:
        from src.sam2_refiner import SAM2MaskRefiner  # noqa: E402

        checkpoint = args.sam_checkpoint if args.sam_checkpoint else sam_cfg["checkpoint"]
        model_cfg = args.sam_config if args.sam_config else sam_cfg["model_cfg"]
        checkpoint = resolve_path(checkpoint, config_path)
        model_cfg = resolve_path(model_cfg, config_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        if not model_cfg.exists():
            raise FileNotFoundError(f"SAM2 model config not found: {model_cfg}")

        print(f"SAM2 config  : {model_cfg}")
        print(f"SAM2 ckpt    : {checkpoint}")
        print(f"Device       : {device}")
        refiner = SAM2MaskRefiner(
            checkpoint=str(checkpoint),
            model_cfg=str(model_cfg),
            device=device,
            multimask_output=sam_cfg.get("multimask_output", True),
            mask_threshold=sam_cfg.get("mask_threshold", 0.0),
            prompt_logit_abs_value=sam_cfg.get("prompt_logit_abs_value", 10.0),
            prompt_size=sam_cfg.get("prompt_size", [256, 256]),
            autocast_dtype=sam_cfg.get("autocast_dtype", "bfloat16"),
        )
    else:
        print("SAM2         : not loaded (no high-coverage samples in this batch)")

    summary_rows = []
    coverage_name = input_cfg.get("coverage_name", "leaf_coverage.txt")

    for index, sample in enumerate(prepared, start=1):
        sample_dir = sample["sample_dir"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        sample_id = sample_dir.name
        mode = sample["mode"]
        coverage = sample["coverage"]
        # sample_output_dir = output_dir / sample_id #Uncomment this line if you want to save the output in a directory named after the sample ID
        sample_output_dir = sample_dir #dele
        output_image_path = sample_output_dir / f"{sample_id}.jpg"
        json_path = sample_output_dir / f"{sample_id}.json"

        print(f"\n[{index}/{len(prepared)}] {sample_id}")
        print(f"  Coverage: {coverage if coverage is not None else 'ERROR'}% -> {mode}")

        if image_path is None:
            summary_rows.append({
                "sample": sample_id, "leaf_coverage_percent": coverage,
                "annotation_mode": mode, "status": "error",
                "output_dir": str(sample_output_dir), "input_components": "",
                "output_polygons": "", "output_points": "", "error": sample["error"],
            })
            continue

        if mode == "error":
            print(f"  ERROR: {sample['coverage_error']}")
            summary_rows.append({
                "sample": sample_id, "leaf_coverage_percent": "",
                "annotation_mode": "error", "status": "error",
                "output_dir": str(sample_output_dir), "input_components": "",
                "output_polygons": "", "output_points": "", "error": sample["coverage_error"],
            })
            continue

        # Middle-coverage samples are intentionally not auto-annotated, but their
        # images are copied so they can be reviewed/annotated manually in LabelMe.
        if mode == "ignore":
            try:
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                if output_cfg.get("copy_image", True):
                    if args.overwrite or not output_image_path.exists():
                        shutil.copy2(image_path, output_image_path)
                else:
                    output_image_path = image_path

                print(f"  IGNORE: copied image for manual review -> {output_image_path}")
                summary_rows.append({
                    "sample": sample_id, "leaf_coverage_percent": coverage,
                    "annotation_mode": "ignore", "status": "ignored_manual_review",
                    "output_dir": str(sample_output_dir), "input_components": "",
                    "output_polygons": 0, "output_points": 0, "error": "",
                })
            except Exception as exc:
                print(f"  ERROR copying ignored image: {type(exc).__name__}: {exc}")
                summary_rows.append({
                    "sample": sample_id, "leaf_coverage_percent": coverage,
                    "annotation_mode": "ignore", "status": "error",
                    "output_dir": str(sample_output_dir), "input_components": "",
                    "output_polygons": "", "output_points": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue

        if json_path.exists() and not args.overwrite:
            print("  SKIP: JSON already exists")
            summary_rows.append({
                "sample": sample_id, "leaf_coverage_percent": coverage,
                "annotation_mode": mode, "status": "skipped",
                "output_dir": str(sample_output_dir), "input_components": "",
                "output_polygons": "", "output_points": "", "error": "",
            })
            continue

        try:
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            if output_cfg.get("copy_image", True):
                shutil.copy2(image_path, output_image_path)
            else:
                output_image_path = image_path

            image_rgb = read_image(image_path)
            original_mask = read_mask(mask_path)
            if image_rgb.shape[:2] != original_mask.shape[:2]:
                raise RuntimeError(
                    f"Image/mask dimensions differ: {image_rgb.shape[:2]} vs {original_mask.shape[:2]}"
                )

            if mode == "negative_point":
                point = get_negative_point(
                    original_mask,
                    image_rgb,
                    coverage_cfg.get("point_source", "mask_centroid"),
                    coverage_cfg.get("empty_mask_fallback", "error"),
                )
                save_labelme_json(
                    output_path=json_path,
                    image_path=output_image_path,
                    polygons=[],
                    label=labelme_cfg.get("label", "leaf"),
                    description=labelme_cfg.get("description", ""),
                    version=labelme_cfg.get("version", "5.11.3"),
                    embed_image_data=labelme_cfg.get("embed_image_data", True),
                    flags={},
                    points=[point],
                    point_label=coverage_cfg.get("negative_label", "negative"),
                    point_description=coverage_cfg.get("negative_description", ""),
                )

                print(f"  Negative point: ({point[0]:.2f}, {point[1]:.2f})")
                summary_rows.append({
                    "sample": sample_id, "leaf_coverage_percent": coverage,
                    "annotation_mode": "negative_point", "status": "ok",
                    "output_dir": str(sample_output_dir), "input_components": "",
                    "output_polygons": 0, "output_points": 1, "error": "",
                })
                continue

            # High coverage: existing CoAtNet2 -> SAM2 -> polygons.
            components = connected_components(
                original_mask,
                min_area=component_cfg.get("min_area", 10000),
                keep_largest_n=component_cfg.get("keep_largest_n", 0),
            )
            print(f"  Existing components: {len(components)}")

            final_mask = np.zeros_like(original_mask, dtype=np.uint8)
            component_logs = []
            if components:
                assert refiner is not None
                refiner.set_image(image_rgb)
                for comp_index, component in enumerate(components):
                    refined, info = refiner.refine_component(
                        component,
                        min_prompt_overlap=refinement_cfg.get("min_prompt_overlap", 0.30),
                    )
                    final_mask = np.maximum(final_mask, refined)
                    component_logs.append({
                        "component_index": comp_index,
                        "prompt_area": int(component.sum()),
                        **info,
                    })

                final_mask = remove_small_components(
                    final_mask,
                    refinement_cfg.get("remove_small_components", 10000),
                )

            polygons = mask_to_polygons(
                final_mask,
                epsilon_ratio=polygon_cfg.get("epsilon_ratio", 0.002),
                min_area=polygon_cfg.get("min_area", 10000),
            )

            save_labelme_json(
                output_path=json_path,
                image_path=output_image_path,
                polygons=polygons,
                label=labelme_cfg.get("label", "leaf"),
                description=labelme_cfg.get("description", ""),
                version=labelme_cfg.get("version", "5.11.3"),
                embed_image_data=labelme_cfg.get("embed_image_data", True),
                flags={},
            )

            if visualization_cfg.get("enabled", False):
                save_mask(sample_output_dir / "sam2_refined_mask.png", final_mask)
                save_overlay(
                    sample_output_dir / "sam2_overlay.jpg",
                    image_rgb,
                    final_mask,
                    alpha=visualization_cfg.get("alpha", 0.45),
                )

            refinement_log = {
                "sample": sample_id,
                "leaf_coverage_percent": coverage,
                "annotation_mode": "sam2_polygon",
                "image": str(image_path),
                "input_mask": str(mask_path),
                "num_input_components": len(components),
                "num_output_polygons": len(polygons),
                "output_json": str(json_path),
                "components": component_logs,
            }
            with (sample_output_dir / "sam2_refinement.json").open("w", encoding="utf-8") as f:
                json.dump(refinement_log, f, indent=2)

            print(f"  Output polygons: {len(polygons)}")
            summary_rows.append({
                "sample": sample_id, "leaf_coverage_percent": coverage,
                "annotation_mode": "sam2_polygon", "status": "ok",
                "output_dir": str(sample_output_dir), "input_components": len(components),
                "output_polygons": len(polygons), "output_points": 0, "error": "",
            })

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            summary_rows.append({
                "sample": sample_id, "leaf_coverage_percent": coverage,
                "annotation_mode": mode, "status": "error",
                "output_dir": str(sample_output_dir), "input_components": "",
                "output_polygons": "", "output_points": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    summary_path = output_dir / "sam2_labelme_summary.csv"
    fieldnames = [
        "sample", "leaf_coverage_percent", "annotation_mode", "status",
        "output_dir", "input_components", "output_polygons", "output_points", "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
