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
        prompt_logit_abs_value=10.0,
        prompt_size=(256, 256),
        autocast_dtype="bfloat16",
    ):
        self.device = device
        self.multimask_output = bool(multimask_output)
        self.mask_threshold = float(mask_threshold)
        self.prompt_logit_abs_value = float(
            prompt_logit_abs_value
        )
        self.prompt_size = tuple(prompt_size)

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

    def _component_to_prompt_logits(self, component):
        h, w = component.shape

        prompt_h, prompt_w = self.prompt_size

        resized = cv2.resize(
            component.astype(np.uint8),
            (prompt_w, prompt_h),
            interpolation=cv2.INTER_NEAREST,
        )

        a = self.prompt_logit_abs_value

        logits = np.where(
            resized > 0,
            a,
            -a,
        ).astype(np.float32)

        return logits[None, ...]

    @staticmethod
    def _overlap(mask, prompt):
        prompt_bool = prompt > 0
        mask_bool = mask > 0

        prompt_area = int(prompt_bool.sum())

        if prompt_area == 0:
            return 0.0

        intersection = int(
            np.logical_and(
                mask_bool,
                prompt_bool,
            ).sum()
        )

        return intersection / prompt_area

    def refine_component(
        self,
        component,
        min_prompt_overlap=0.30,
    ):
        prompt_logits = self._component_to_prompt_logits(
            component
        )

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
                    mask_input=prompt_logits,
                    multimask_output=self.multimask_output,
                )
        else:
            masks, scores, _ = self.predictor.predict(
                mask_input=prompt_logits,
                multimask_output=self.multimask_output,
            )

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)

        if masks.ndim == 2:
            masks = masks[None, ...]

        prompt = component

        candidates = []

        for i in range(len(masks)):
            candidate = masks[i]

            if candidate.shape != component.shape:
                candidate = cv2.resize(
                    candidate.astype(np.uint8),
                    (
                        component.shape[1],
                        component.shape[0],
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

            candidate = (
                candidate > self.mask_threshold
            ).astype(np.uint8)

            overlap = self._overlap(
                candidate,
                prompt,
            )

            candidates.append(
                {
                    "index": i,
                    "mask": candidate,
                    "score": float(scores[i]),
                    "overlap": float(overlap),
                }
            )

        accepted = [
            item
            for item in candidates
            if item["overlap"] >= float(
                min_prompt_overlap
            )
        ]

        if accepted:
            selected = max(
                accepted,
                key=lambda item: (
                    item["score"],
                    item["overlap"],
                ),
            )
        else:
            selected = max(
                candidates,
                key=lambda item: (
                    item["score"],
                    item["overlap"],
                ),
            )

        return selected["mask"], {
            "selected_index": int(
                selected["index"]
            ),
            "selected_score": float(
                selected["score"]
            ),
            "selected_overlap": float(
                selected["overlap"]
            ),
            "num_candidates": len(candidates),
            "num_accepted": len(accepted),
        }
