# cross_dataset_eval.py
#
# Evaluates a trained HD-MixNet checkpoint on every dataset listed in
# Config.CROSS_DATASETS, skipping any that are not on disk. This addresses
# the reviewer's note that the thesis only used Kvasir-SEG.
#
# Dataset layout expected per root:
#     <root>/images/*.png|jpg
#     <root>/masks/*.png|jpg
#
# (Some public splits use 'Original' and 'Ground Truth' subfolders - rename
#  those to 'images' and 'masks' or pass --img-dir / --mask-dir explicitly
#  for one-off evaluations.)
#
# Example:
#     python cross_dataset_eval.py --path checkpoints/best_dice_model.pth \
#         --output results/cross_dataset.json

import argparse
import json
import os

import torch

from config import Config
from evaluate import evaluate, _ci95   # reuses the per-image-metric pipeline


def discover_datasets(config):
    """Return list of (name, img_dir, mask_dir) for datasets that exist."""
    found = []
    missing = []
    for name, root in config.CROSS_DATASETS.items():
        img_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'masks')
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            found.append((name, img_dir, mask_dir))
        else:
            missing.append((name, root))
    return found, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoints/best_dice_model.pth')
    parser.add_argument('--use-tta', action='store_true')
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--measure-speed', action='store_true')
    parser.add_argument('--output', type=str, default='./results/cross_dataset.json')
    args = parser.parse_args()

    config = Config()
    found, missing = discover_datasets(config)

    print("Cross-dataset evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.path}")
    print(f"Datasets found: {[n for n, _, _ in found]}")
    if missing:
        print(f"Datasets missing (skipped): {[n for n, _ in missing]}")
    print("=" * 60)

    all_results = {
        'checkpoint': args.path,
        'use_tta': bool(args.use_tta),
        'datasets': {},
        'datasets_missing': [n for n, _ in missing],
    }

    for name, img_dir, mask_dir in found:
        print(f"\n--- {name} ---")
        results = evaluate(
            args.path,
            use_tta=args.use_tta,
            img_size=args.img_size,
            img_dir=img_dir,
            mask_dir=mask_dir,
            measure_speed=False,
        )
        all_results['datasets'][name] = {r['metric']: r for r in results}

    if args.measure_speed:
        # Only need one speed measurement, not one per dataset.
        from evaluate import measure_fps
        from Models.hd_mixnet import HD_MixNet
        from Utils.inference import load_checkpoint
        raw = torch.load(args.path, map_location='cpu')
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            if 'branch_mode' in raw:    config.BRANCH_MODE = raw['branch_mode']
            if 'use_bamf' in raw:       config.USE_BAMF = bool(raw['use_bamf'])
            if 'use_edge_sup' in raw:   config.USE_EDGE_SUP = bool(raw['use_edge_sup'])
        model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
        load_checkpoint(model, args.path, config.DEVICE)
        all_results['inference_speed'] = measure_fps(model, config)
        print("\nInference speed:", all_results['inference_speed'])

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results: {args.output}")


if __name__ == "__main__":
    main()
