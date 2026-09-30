from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2MaskRefiner:
    def __init__(
        self,
        checkpoint,
        model_cfg,
        device="cuda",
        multimask_output=True,
        mask_threshold=0.0,
        autocast_dtype="bfloat16",
    ):
        self.device = device
        self.multimask_output = bool(multimask_output)
        self.mask_threshold = float(mask_threshold)
        self.autocast_dtype = self._resolve_dtype(
            autocast_dtype
        )

        self.model = build_sam2(
            model_cfg,
            checkpoint,
            device=device,
        )

        self.predictor = SAM2ImagePredictor(
            self.model
        )

    @staticmethod
    def _resolve_dtype(value):
        if value is None:
            return None

        if isinstance(value, torch.dtype):
            return value

        name = str(value).lower()

        if name in {"bfloat16", "bf16", "torch.bfloat16"}:
            return torch.bfloat16

        if name in {"float16", "fp16", "half", "torch.float16"}:
            return torch.float16

        if name in {"float32", "fp32", "torch.float32"}:
            return torch.float32

        raise ValueError(
            f"Unsupported autocast dtype: {value}"
        )

    def set_image(self, image_rgb):
        self.predictor.set_image(image_rgb)

    @staticmethod
    def _component_centroid(component):
        moments = cv2.moments((component > 0).astype(np.uint8))
        if moments["m00"] == 0:
            raise ValueError("Cannot prompt SAM2 with an empty component")
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    def refine_component(self, component):
        point_x, point_y = self._component_centroid(component)
        point_coords = np.array([[point_x, point_y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        autocast_enabled = (
            self.device.startswith("cuda")
            and self.autocast_dtype is not None
            and torch.cuda.is_available()
        )

        if autocast_enabled:
            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
            ):
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=self.multimask_output,
                )
        else:
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=self.multimask_output,
            )

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)

        if masks.ndim == 2:
            masks = masks[None, ...]

        # SAM2 returns candidate masks in predictor order; select its highest
        # scoring candidate for this positive point prompt.
        selected_index = int(np.argmax(scores))
        selected_mask = masks[selected_index]
        if selected_mask.shape != component.shape:
            selected_mask = cv2.resize(
                selected_mask.astype(np.uint8),
                (component.shape[1], component.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        selected_mask = (selected_mask > self.mask_threshold).astype(np.uint8)

        return selected_mask, {
            "prompt_point_x": point_x,
            "prompt_point_y": point_y,
            "selected_index": selected_index,
            "selected_score": float(scores[selected_index]),
            "num_candidates": len(masks),
        }
