from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm


class CoAtNet2Backbone(nn.Module):
    """
    extract only student.backbone.backbone.*

    """

    def __init__(
        self,
        checkpoint_path: str | None,
        model_name: str = "coatnet_2_rw_224",
        out_indices=(1, 2, 3, 4),
        expected_channels=(128, 256, 512, 1024),
        pretrained: bool = False,
    ):
        super().__init__()

        self.out_indices = tuple(out_indices)
        self.expected_channels = tuple(expected_channels)

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        self.feature_info = self.encoder.feature_info
        self.channels = list(self.feature_info.channels())
        self.reductions = list(self.feature_info.reduction())

        if len(self.channels) != len(self.expected_channels):
            raise RuntimeError(
                f"Expected {len(self.expected_channels)} feature maps, "
                f"but timm returned {len(self.channels)}: {self.channels}"
            )

        if self.channels != list(self.expected_channels):
            raise RuntimeError(
                "Backbone channel mismatch.\n"
                f"Expected checkpoint channels: {self.expected_channels}\n"
                f"timm returned: {self.channels}\n"
                "Use a compatible timm version/model implementation."
            )

        if checkpoint_path:
            self.load_foundation_checkpoint(checkpoint_path)

    @staticmethod
    def _map_checkpoint_key(key):
        """
        Convert foundation checkpoint keys to our model's
        module naming convention.

        Source:
            stages.0.blocks.0.pre_norm.weight

        Model:
            stages_0.blocks.0.pre_norm.weight
        """

        if key.startswith("stages.0."):
            key = key.replace(
                "stages.0.",
                "stages_0.",
                1,
            )

        elif key.startswith("stages.1."):
            key = key.replace(
                "stages.1.",
                "stages_1.",
                1,
            )

        elif key.startswith("stages.2."):
            key = key.replace(
                "stages.2.",
                "stages_2.",
                1,
            )

        elif key.startswith("stages.3."):
            key = key.replace(
                "stages.3.",
                "stages_3.",
                1,
            )

        return key

    def load_foundation_checkpoint(self, checkpoint_path):
        print(f"\nLoading foundation checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        # ---------------------------------------------------------
        # checkpoint["student"]
        #     ["backbone.backbone.stem.norm1.weight"]
        #     ["backbone.backbone.stem.norm1.bias"]
        #     ["backbone.backbone.stem.conv1.weight"]
        #     ...
        # ---------------------------------------------------------

        if "student" not in checkpoint:
            raise RuntimeError(
                "Could not find 'student' in foundation checkpoint."
            )

        student = checkpoint["student"]

        if not isinstance(student, dict):
            raise RuntimeError(
                f"checkpoint['student'] must be a dict, "
                f"got {type(student)}"
            )

        prefix = "backbone.backbone."

        source_state = {}

        for key, value in student.items():

            if not torch.is_tensor(value):
                continue

            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_key = self._map_checkpoint_key(new_key)
                source_state[new_key] = value

        if not source_state:
            raise RuntimeError(
                "Could not find keys beginning with "
                "'backbone.backbone.' inside checkpoint['student']."
            )

        print(
            f"Found {len(source_state)} backbone tensors "
            f"in foundation checkpoint."
        )

        # ---------------------------------------------------------
        # Match against timm model
        # ---------------------------------------------------------

        model_state = self.encoder.state_dict()

        compatible = {}
        shape_mismatches = []
        unused = []

        for i, (key, value) in enumerate(source_state.items()):
            print(i, key)

            if key not in model_state:
                unused.append(key)
                continue

            if model_state[key].shape != value.shape:
                shape_mismatches.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(model_state[key].shape),
                    )
                )
                continue

            compatible[key] = value

        print("\nFoundation checkpoint compatibility:")
        print(f"  Source backbone tensors : {len(source_state)}")
        print(f"  Compatible tensors      : {len(compatible)}")
        print(f"  Shape mismatches        : {len(shape_mismatches)}")
        print(f"  Unused source tensors   : {len(unused)}")
        print(f"  Model tensors           : {len(model_state)}")

        if shape_mismatches:
            print("\nFirst shape mismatches:")
            for key, src_shape, dst_shape in shape_mismatches[:20]:
                print(
                    f"  {key}: "
                    f"checkpoint={src_shape}, "
                    f"model={dst_shape}"
                )

        if unused:
            print("\nFirst unused checkpoint keys:")
            for key in unused[:20]:
                print(f"  {key}")

        missing = [
            key for key in model_state
            if key not in compatible
        ]

        print(f"  Missing model tensors    : {len(missing)}")

        if missing:
            print("\nFirst missing model keys:")
            for key in missing[:20]:
                print(f"  {key}")

        # ---------------------------------------------------------
        # Coverage
        # ---------------------------------------------------------

        coverage = len(compatible) / len(model_state)

        print(f"\nModel tensor coverage: {coverage:.2%}")

        if len(compatible) == 0:
            raise RuntimeError(
                "No compatible backbone tensors were found."
            )

        if coverage < 0.50:
            raise RuntimeError(
                f"Foundation checkpoint coverage is only "
                f"{coverage:.2%}. "
                f"Refusing to continue because the checkpoint "
                f"and timm model appear incompatible."
            )

        # ---------------------------------------------------------
        # Load compatible tensors
        # ---------------------------------------------------------

        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            compatible,
            strict=False
        )

        print("\nCheckpoint loaded successfully.")
        print(f"  Missing keys after load    : {len(missing_keys)}")
        print(f"  Unexpected keys after load : {len(unexpected_keys)}")
        def forward(self, x):
            outputs = self.encoder(x)
            return {
                f"f{i + 1}": feature
                for i, feature in enumerate(outputs)
            }

        def freeze(self):
            for p in self.parameters():
                p.requires_grad = False

        def unfreeze(self):
            for p in self.parameters():
                p.requires_grad = True
