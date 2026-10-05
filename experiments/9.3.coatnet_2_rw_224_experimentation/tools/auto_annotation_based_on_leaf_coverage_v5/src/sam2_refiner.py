from __future__ import annotations

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
        target_pixels_per_prompt=5000,
        max_points_per_region=16,
        blur_filter_enabled=True,
        blur_min_laplacian_variance=20.0,
        blur_patch_size=31,
    ):
        self.device = device
        self.multimask_output = bool(multimask_output)
        self.mask_threshold = float(mask_threshold)
        self.target_pixels_per_prompt = max(1, int(target_pixels_per_prompt))
        self.max_points_per_region = max(1, int(max_points_per_region))
        self.blur_filter_enabled = bool(blur_filter_enabled)
        self.blur_min_laplacian_variance = float(blur_min_laplacian_variance)
        self.blur_patch_size = max(3, int(blur_patch_size) | 1)
        self.image_gray = None
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
        self.image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def _sharpness_at(self, point_x, point_y):
        half = self.blur_patch_size // 2
        height, width = self.image_gray.shape
        x, y = int(round(point_x)), int(round(point_y))
        patch = self.image_gray[
            max(0, y - half):min(height, y + half + 1),
            max(0, x - half):min(width, x + half + 1),
        ]
        return float(cv2.Laplacian(patch, cv2.CV_64F).var())

    def _component_centroids(self, component):
        ys, xs = np.nonzero(component)
        area = len(xs)
        if area == 0:
            raise ValueError("Cannot prompt SAM2 with an empty component")

        count = min(
            self.max_points_per_region,
            max(1, int(np.ceil(area / self.target_pixels_per_prompt))),
        )
        if count == 1:
            center_x, center_y = xs.mean(), ys.mean()
            nearest = np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2)
            return [(float(xs[nearest]), float(ys[nearest]))]

        # Cluster foreground pixels so each prompt lies inside the predicted
        # region and covers a different part of large or merged regions.
        samples = np.column_stack((xs, ys)).astype(np.float32)
        cv2.setRNGSeed(0)
        _, _, centers = cv2.kmeans(
            samples,
            count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        centroids = []
        for center_x, center_y in centers:
            nearest = np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2)
            centroids.append((float(xs[nearest]), float(ys[nearest])))
        return centroids

    def refine_component(self, component):
        centroids = self._component_centroids(component)
        sharpness_scores = [self._sharpness_at(x, y) for x, y in centroids]
        accepted = [
            (point, score)
            for point, score in zip(centroids, sharpness_scores)
            if not self.blur_filter_enabled
            or score >= self.blur_min_laplacian_variance
        ]
        accepted_centroids = [point for point, _ in accepted]
        rejected_count = len(centroids) - len(accepted_centroids)

        if not accepted_centroids:
            return np.zeros_like(component, dtype=np.uint8), {
                "prompt_points": [],
                "num_prompts": 0,
                "num_blur_rejected": rejected_count,
                "prompt_sharpness": [
                    {"x": x, "y": y, "laplacian_variance": score, "accepted": False}
                    for (x, y), score in zip(centroids, sharpness_scores)
                ],
                "num_unique_masks": 0,
                "selected_scores": [],
                "num_candidates_per_prompt": [],
            }

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
                prompt_results = [
                    self.predictor.predict(
                        point_coords=np.array([[x, y]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=self.multimask_output,
                    )
                    for x, y in accepted_centroids
                ]
        else:
            prompt_results = [
                self.predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=self.multimask_output,
                )
                for x, y in accepted_centroids
            ]

        combined = np.zeros_like(component, dtype=np.uint8)
        selected_scores = []
        unique_masks = set()
        candidate_counts = []
        for (point_x, point_y), (masks, scores, _) in zip(accepted_centroids, prompt_results):
            masks = np.asarray(masks)
            scores = np.asarray(scores).reshape(-1)
            if masks.ndim == 2:
                masks = masks[None, ...]
            selected_index = int(np.argmax(scores))
            selected_mask = masks[selected_index]
            if selected_mask.shape != component.shape:
                selected_mask = cv2.resize(
                    selected_mask.astype(np.uint8),
                    (component.shape[1], component.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            selected_mask = (selected_mask > self.mask_threshold).astype(np.uint8)
            unique_masks.add(selected_mask.tobytes())
            combined = np.maximum(combined, selected_mask)
            selected_scores.append(float(scores[selected_index]))
            candidate_counts.append(len(masks))

        return combined, {
            "prompt_points": [{"x": x, "y": y} for x, y in accepted_centroids],
            "num_prompts": len(accepted_centroids),
            "num_blur_rejected": rejected_count,
            "prompt_sharpness": [
                {
                    "x": x,
                    "y": y,
                    "laplacian_variance": score,
                    "accepted": score >= self.blur_min_laplacian_variance
                    or not self.blur_filter_enabled,
                }
                for (x, y), score in zip(centroids, sharpness_scores)
            ],
            "num_unique_masks": len(unique_masks),
            "selected_scores": selected_scores,
            "num_candidates_per_prompt": candidate_counts,
        }
