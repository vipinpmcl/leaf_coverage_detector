from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running:
# python scripts/refine_batch.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components import (  # noqa: E402
    connected_components,
    remove_small_components,
)
from src.input_loader import (  # noqa: E402
    discover_prediction_samples,
    read_image,
    read_mask,
)
from src.labelme import (  # noqa: E402
    save_labelme_json,
)
from src.polygons import mask_to_polygons  # noqa: E402
from src.sam2_refiner import SAM2MaskRefiner  # noqa: E402
from src.visualization import (  # noqa: E402
    save_mask,
    save_overlay,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value, config_path):
    path = Path(value)

    if path.is_absolute():
        return path

    # Prefer project-root-relative paths.
    project_path = PROJECT_ROOT / path

    if project_path.exists():
        return project_path

    # Also support paths relative to config file.
    return Path(config_path).resolve().parent / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refine existing CoAtNet2 leaf masks "
            "with SAM2 and create LabelMe JSON."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means process all samples.",
    )

    parser.add_argument(
        "--sam-checkpoint",
        default=None,
    )

    parser.add_argument(
        "--model-cfg",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    input_cfg = cfg.get("input", {})
    sam_cfg = cfg.get("sam2", {})
    component_cfg = cfg.get("components", {})
    refinement_cfg = cfg.get("refinement", {})
    polygon_cfg = cfg.get("polygon", {})
    visualization_cfg = cfg.get(
        "visualization",
        {},
    )
    labelme_cfg = cfg.get("labelme", {})

    device = cfg.get("device", "cuda")

    checkpoint = (
        args.sam_checkpoint
        if args.sam_checkpoint
        else sam_cfg["checkpoint"]
    )

    checkpoint = resolve_path(
        checkpoint,
        config_path,
    )

    model_cfg = (
        args.model_cfg
        if args.model_cfg
        else sam_cfg["model_cfg"]
    )

    model_cfg = resolve_path(
        model_cfg,
        config_path,
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}"
        )

    if not model_cfg.exists():
        raise FileNotFoundError(
            f"SAM2 model config not found: {model_cfg}"
        )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input dir    : {Path(args.input_dir).resolve()}")
    print(f"SAM2 config  : {model_cfg}")
    print(f"SAM2 ckpt    : {checkpoint}")
    print(f"Device       : {device}")

    samples = list(
        discover_prediction_samples(
            args.input_dir,
            original_name=input_cfg.get(
                "original_name",
                "original.jpg",
            ),
            mask_name=input_cfg.get(
                "mask_name",
                "mask.png",
            ),
            skip_directories=input_cfg.get(
                "skip_directories",
                ["clusters_output"],
            ),
        )
    )

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Samples      : {len(samples)}")

    if not samples:
        print("No valid samples found.")
        return

    refiner = SAM2MaskRefiner(
        checkpoint=str(checkpoint),
        model_cfg=str(model_cfg),
        device=device,
        multimask_output=sam_cfg.get(
            "multimask_output",
            True,
        ),
        mask_threshold=sam_cfg.get(
            "mask_threshold",
            0.0,
        ),
        prompt_logit_abs_value=sam_cfg.get(
            "prompt_logit_abs_value",
            10.0,
        ),
        prompt_size=sam_cfg.get(
            "prompt_size",
            [256, 256],
        ),
        autocast_dtype=sam_cfg.get(
            "autocast_dtype",
            "bfloat16",
        ),
    )

    summary_rows = []

    for index, sample in enumerate(samples, start=1):
        sample_dir = sample["sample_dir"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        sample_id = sample_dir.name

        sample_image_path = sample_dir / f"{sample_id}.jpg"

        if not sample_image_path.exists():
            import shutil
            shutil.copy2(image_path, sample_image_path)
        
        json_path = sample_dir / f"{sample_id}.json"

        print(
            f"\n[{index}/{len(samples)}] "
            f"{sample_id}"
        )

        if image_path is None:
            print(f"  ERROR: {sample['error']}")
            summary_rows.append(
                {
                    "sample": sample_id,
                    "status": "error",
                    "error": sample["error"],
                }
            )
            continue

        if json_path.exists() and not args.overwrite:
            print("  SKIP: JSON already exists")
            summary_rows.append(
                {
                    "sample": sample_id,
                    "status": "skipped",
                    "error": "",
                }
            )
            continue

        try:
            image_rgb = read_image(image_path)
            original_mask = read_mask(mask_path)

            if (
                image_rgb.shape[:2]
                != original_mask.shape[:2]
            ):
                raise RuntimeError(
                    "Image/mask dimensions differ: "
                    f"{image_rgb.shape[:2]} vs "
                    f"{original_mask.shape[:2]}"
                )

            components = connected_components(
                original_mask,
                min_area=component_cfg.get(
                    "min_area",
                    10000,
                ),
                keep_largest_n=component_cfg.get(
                    "keep_largest_n",
                    0,
                ),
            )

            print(
                f"  Existing components: "
                f"{len(components)}"
            )

            if not components:
                final_mask = np.zeros_like(
                    original_mask,
                    dtype=np.uint8,
                )
                polygons = []
                component_logs = []
            else:
                refiner.set_image(image_rgb)

                final_mask = np.zeros_like(
                    original_mask,
                    dtype=np.uint8,
                )

                component_logs = []

                for comp_index, component in enumerate(
                    components
                ):
                    refined, info = (
                        refiner.refine_component(
                            component,
                            min_prompt_overlap=refinement_cfg.get(
                                "min_prompt_overlap",
                                0.30,
                            ),
                        )
                    )

                    final_mask = np.maximum(
                        final_mask,
                        refined,
                    )

                    component_logs.append(
                        {
                            "component_index": comp_index,
                            "prompt_area": int(
                                component.sum()
                            ),
                            **info,
                        }
                    )

                final_mask = (
                    remove_small_components(
                        final_mask,
                        refinement_cfg.get(
                            "remove_small_components",
                            10000,
                        ),
                    )
                )

                polygons = mask_to_polygons(
                    final_mask,
                    epsilon_ratio=polygon_cfg.get(
                        "epsilon_ratio",
                        0.002,
                    ),
                    min_area=polygon_cfg.get(
                        "min_area",
                        10000,
                    ),
                )

            save_labelme_json(
                output_path=json_path,
                image_path=sample_image_path,
                polygons=polygons,
                label=labelme_cfg.get(
                    "label",
                    "leaf",
                ),
                description=labelme_cfg.get(
                    "description",
                    "",
                ),
                version=labelme_cfg.get(
                    "version",
                    "5.11.3",
                ),
                embed_image_data=labelme_cfg.get(
                    "embed_image_data",
                    True,
                ),
                flags={},
            )

            if visualization_cfg.get(
                "enabled",
                True,
            ):
                save_mask(
                    sample_dir / "sam2_refined_mask.png",
                    final_mask,
                )

                save_overlay(
                    sample_dir / "sam2_overlay.jpg",
                    image_rgb,
                    final_mask,
                    alpha=visualization_cfg.get(
                        "alpha",
                        0.45,
                    ),
                )

            refinement_log = {
                "sample": sample_id,
                "image": str(image_path),
                "input_mask": str(mask_path),
                "num_input_components": len(
                    components
                ),
                "num_output_polygons": len(
                    polygons
                ),
                "output_json": str(json_path),
                "components": component_logs,
            }

            with (
                sample_dir
                / "sam2_refinement.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    refinement_log,
                    f,
                    indent=2,
                )

            print(
                f"  Output polygons: "
                f"{len(polygons)}"
            )
            print(
                f"  JSON: {json_path}"
            )

            summary_rows.append(
                {
                    "sample": sample_id,
                    "status": "ok",
                    "input_components": len(
                        components
                    ),
                    "output_polygons": len(
                        polygons
                    ),
                    "error": "",
                }
            )

        except Exception as exc:
            print(
                f"  ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            summary_rows.append(
                {
                    "sample": sample_id,
                    "status": "error",
                    "input_components": "",
                    "output_polygons": "",
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    summary_path = (
        Path(args.input_dir).resolve()
        / "sam2_labelme_summary.csv"
    )

    fieldnames = [
        "sample",
        "status",
        "input_components",
        "output_polygons",
        "error",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"\nSummary: {summary_path}"
    )


if __name__ == "__main__":
    main()
