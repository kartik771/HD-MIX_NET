# evaluate.py
#
# Adds:
#   - per-image metrics collection + mean / std / 95% CI report
#   - inference-time and FPS measurement (CPU or CUDA), with warmup
#   - reads the checkpoint's recorded BRANCH_MODE / USE_BAMF / USE_EDGE_SUP
#     and rebuilds the model with those flags so that ablation checkpoints
#     load cleanly without manual config edits.
#
# Cross-dataset evaluation is in cross_dataset_eval.py.

import argparse
import os
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import load_checkpoint, predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.metrics import dice_coef, iou_score, hausdorff_95
from config import Config


def _apply_ablation_from_checkpoint(config, meta):
    """If the checkpoint records ablation flags, mirror them into the Config
    used to build the model. This avoids a class of frustrating bugs where
    one trains a 'cnn_only' checkpoint and then loads it with the default
    'both' config and a key-mismatch error appears."""
    if not isinstance(meta, dict):
        return
    if 'branch_mode' in meta:
        config.BRANCH_MODE = meta['branch_mode']
    if 'use_bamf' in meta:
        config.USE_BAMF = bool(meta['use_bamf'])
    if 'use_edge_sup' in meta:
        config.USE_EDGE_SUP = bool(meta['use_edge_sup'])


def _ci95(values):
    if len(values) <= 1:
        return 0.0
    a = np.asarray(values, dtype=np.float64)
    sem = a.std(ddof=1) / math.sqrt(len(a))
    # z=1.96 for 95% CI; using normal approximation since N is moderate.
    return 1.96 * sem


def measure_fps(model, config, n_warmup=10, n_iters=50):
    """Time forward passes on a single image of the configured inference size."""
    model.eval()
    device = config.DEVICE
    img_size = config.INFERENCE_IMG_SIZE
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_img = (elapsed / n_iters) * 1000.0
    fps = n_iters / elapsed
    return {'ms_per_image': ms_per_img, 'fps': fps, 'n_iters': n_iters, 'device': str(device)}


def evaluate(model_path, use_tta=False, batch_size=None, img_size=None, measure_speed=False,
             img_dir=None, mask_dir=None):
    config = Config()
    if batch_size is not None:
        config.INFERENCE_BATCH_SIZE = batch_size
    if img_size is not None:
        config.INFERENCE_IMG_SIZE = img_size
    if img_dir is None:
        img_dir = config.TRAIN_IMG_DIR
    if mask_dir is None:
        mask_dir = config.TRAIN_MASK_DIR

    # Build model with default config flags, then patch flags from checkpoint
    # metadata BEFORE constructing the network.
    raw_meta = torch.load(model_path, map_location='cpu')
    if isinstance(raw_meta, dict) and 'model_state_dict' in raw_meta:
        _apply_ablation_from_checkpoint(config, raw_meta)

    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    meta = load_checkpoint(model, model_path, config.DEVICE)
    model.eval()
    threshold = float(meta.get('threshold', config.DEFAULT_THRESHOLD))

    ds = KvasirDataset(img_dir=img_dir, mask_dir=mask_dir,
                       transforms=get_transforms('test', config.INFERENCE_IMG_SIZE))
    loader = DataLoader(ds, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)

    print(f"Evaluating {model_path}")
    print(f"  Branch mode: {getattr(config, 'BRANCH_MODE', 'both')}  "
          f"BAMF: {config.USE_BAMF}  Edge sup: {config.USE_EDGE_SUP}")
    print(f"  Dataset: {img_dir}  ({len(ds)} images)")
    print(f"  threshold={threshold:.2f}  TTA={'on' if use_tta else 'off'}  "
          f"batch={config.INFERENCE_BATCH_SIZE}  img_size={config.INFERENCE_IMG_SIZE}")

    dice_vals, iou_vals, hd_vals = [], [], []

    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            image = image.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            prob = predict_probabilities(model, image, use_tta=use_tta)
            pred = probabilities_to_mask(prob, threshold, config=config)

            for b in range(pred.shape[0]):
                p = pred[b:b + 1]
                m = mask[b:b + 1]
                dice_vals.append(dice_coef(p, m, from_logits=False))
                iou_vals.append(iou_score(p, m, from_logits=False))
                hd_vals.append(hausdorff_95(p, m, from_logits=False))

            if i % 50 == 0:
                last_d = dice_vals[-1] if dice_vals else 0
                last_i = iou_vals[-1] if iou_vals else 0
                last_h = hd_vals[-1] if hd_vals else 0
                print(f"  batch {i}: dice={last_d:.4f} iou={last_i:.4f} hd95={last_h:.2f}")

    def stats(name, vals):
        a = np.asarray(vals, dtype=np.float64)
        return {
            'metric': name,
            'mean': float(a.mean()),
            'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
            'ci95': float(_ci95(vals)),
            'min': float(a.min()),
            'max': float(a.max()),
            'median': float(np.median(a)),
            'n': len(a),
        }

    results = [stats('Dice', dice_vals), stats('IoU', iou_vals), stats('HD95', hd_vals)]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'metric':<8} {'mean':>8} {'±95%CI':>10} {'std':>8} {'median':>8} {'min':>8} {'max':>8}  n")
    for r in results:
        print(f"{r['metric']:<8} {r['mean']:>8.4f} {r['ci95']:>10.4f} {r['std']:>8.4f} "
              f"{r['median']:>8.4f} {r['min']:>8.4f} {r['max']:>8.4f}  {r['n']}")
    print("=" * 60)

    if measure_speed:
        fps = measure_fps(model, config)
        print(f"\nInference speed @ {config.INFERENCE_IMG_SIZE}x{config.INFERENCE_IMG_SIZE}, batch=1:")
        print(f"  {fps['ms_per_image']:.2f} ms/image   ({fps['fps']:.2f} FPS)   "
              f"over {fps['n_iters']} iters on {fps['device']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoints/best_dice_model.pth')
    parser.add_argument('--use-tta', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--measure-speed', action='store_true',
                        help='Time forward passes and report ms/image + FPS')
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Override images directory (for cross-dataset eval)')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Override masks directory')
    args = parser.parse_args()

    evaluate(
        args.path,
        use_tta=args.use_tta,
        batch_size=args.batch_size,
        img_size=args.img_size,
        measure_speed=args.measure_speed,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
    )
