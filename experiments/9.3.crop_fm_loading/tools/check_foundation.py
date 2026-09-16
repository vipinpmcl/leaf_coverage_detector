import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.backbone import CoAtNet2Backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    _ = CoAtNet2Backbone(
        checkpoint_path=args.checkpoint,
        model_name="coatnet_2_rw_224",
        out_indices=(1, 2, 3, 4),
        expected_channels=(128, 256, 512, 1024),
        pretrained=False,
    )

    print("\nPASSED.")


if __name__ == "__main__":
    main()
