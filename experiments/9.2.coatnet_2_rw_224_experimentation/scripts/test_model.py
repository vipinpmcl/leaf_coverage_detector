import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from models import CoAtNetLeafDetector

def main():
    parser = argparse.ArgumentParser(description="Inspect CoAtNet-2 leaf segmentation model.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    model = CoAtNetLeafDetector(pretrained=args.pretrained).eval()
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        logits, raw, adapted = model(x, return_features=True)

    print("=== CoAtNet-2 Leaf Detector ===")
    print("Input:", tuple(x.shape))
    print("Total parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Feature channels:", model.backbone.channels)
    print("Feature reductions:", model.backbone.reductions)
    print("\nRaw CoAtNet features:")
    for name, t in raw.items():
        print(f"  {name}: {tuple(t.shape)}")
    print("\nAdapted features:")
    for i, t in enumerate(adapted, 1):
        print(f"  f{i}: {tuple(t.shape)}")
    print("\nMask logits:", tuple(logits.shape))

if __name__ == "__main__":
    main()
