# summarize_ablations.py
#
# Walks the checkpoints/ablations/ tree, evaluates each best_dice_model.pth
# on the held-out validation split recorded in that run's
# run_metadata.json, aggregates per-seed numbers into mean ± 95% CI, and
# writes a single results table.
#
# Outputs:
#   results/ablation_table.csv   (one row per (variant, seed))
#   results/ablation_summary.csv (one row per variant: mean, std, 95% CI, n_seeds)
#   results/ablation_summary.md  (human-readable summary table)

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import load_checkpoint, predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.metrics import dice_coef, iou_score, hausdorff_95


SEED_RE = re.compile(r'_seed(\d+)$')


def _ci95(values):
    if len(values) <= 1:
        return 0.0
    a = np.asarray(values, dtype=np.float64)
    return 1.96 * a.std(ddof=1) / math.sqrt(len(a))


def evaluate_checkpoint(ckpt_path, run_meta, base_config):
    config = Config()
    raw = torch.load(ckpt_path, map_location='cpu')
    if 'branch_mode' in raw:    config.BRANCH_MODE = raw['branch_mode']
    if 'use_bamf' in raw:       config.USE_BAMF = bool(raw['use_bamf'])
    if 'use_edge_sup' in raw:   config.USE_EDGE_SUP = bool(raw['use_edge_sup'])

    val_ids = run_meta.get('val_ids', [])
    if not val_ids:
        print(f"[skip] {ckpt_path}: no val_ids recorded in run_metadata.json")
        return None

    ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR, mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('test', config.INFERENCE_IMG_SIZE), file_names=val_ids,
    )
    loader = DataLoader(ds, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)

    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    meta = load_checkpoint(model, ckpt_path, config.DEVICE)
    model.eval()
    threshold = float(meta.get('threshold', config.DEFAULT_THRESHOLD))

    dvals, ivals, hvals = [], [], []
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(config.DEVICE); mask = mask.to(config.DEVICE)
            prob = predict_probabilities(model, image, use_tta=False)
            pred = probabilities_to_mask(prob, threshold, config=config)
            for b in range(pred.shape[0]):
                p = pred[b:b + 1]; m = mask[b:b + 1]
                dvals.append(dice_coef(p, m, from_logits=False))
                ivals.append(iou_score(p, m, from_logits=False))
                hvals.append(hausdorff_95(p, m, from_logits=False))

    return {
        'dice_mean': float(np.mean(dvals)),
        'iou_mean':  float(np.mean(ivals)),
        'hd95_mean': float(np.mean(hvals)),
        'dice_ci95': float(_ci95(dvals)),
        'iou_ci95':  float(_ci95(ivals)),
        'hd95_ci95': float(_ci95(hvals)),
        'n_images':  len(dvals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./checkpoints/ablations')
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Walk one level deep: <root>/<variant_seedN>/best_dice_model.pth
    per_run = []  # list of (variant, seed, metrics_dict)
    for run_dir in sorted(glob.glob(os.path.join(args.root, '*'))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        m = SEED_RE.search(name)
        if not m:
            print(f"[skip] {run_dir}: name does not end in _seed<N>")
            continue
        variant = name[: m.start()]
        seed = int(m.group(1))

        meta_path = os.path.join(run_dir, 'run_metadata.json')
        ckpt_path = os.path.join(run_dir, 'best_dice_model.pth')
        if not (os.path.exists(meta_path) and os.path.exists(ckpt_path)):
            print(f"[skip] {run_dir}: missing run_metadata.json or best_dice_model.pth")
            continue

        with open(meta_path, 'r') as f:
            run_meta = json.load(f)

        print(f"Evaluating {variant} seed={seed} ...")
        res = evaluate_checkpoint(ckpt_path, run_meta, Config())
        if res is None:
            continue
        res['variant'] = variant
        res['seed'] = seed
        per_run.append(res)

    if not per_run:
        print("No completed ablation runs found.")
        return

    # Per-run CSV
    csv_path = os.path.join(args.out_dir, 'ablation_table.csv')
    fields = ['variant', 'seed', 'n_images',
              'dice_mean', 'dice_ci95',
              'iou_mean', 'iou_ci95',
              'hd95_mean', 'hd95_ci95']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r.get(k) for k in fields})

    # Per-variant summary across seeds
    by_variant = defaultdict(list)
    for r in per_run:
        by_variant[r['variant']].append(r)

    summary_rows = []
    for variant, runs in by_variant.items():
        dices = [r['dice_mean'] for r in runs]
        ious  = [r['iou_mean']  for r in runs]
        hds   = [r['hd95_mean'] for r in runs]
        summary_rows.append({
            'variant': variant,
            'n_seeds': len(runs),
            'dice_mean': float(np.mean(dices)),
            'dice_seed_ci95': float(_ci95(dices)),
            'iou_mean': float(np.mean(ious)),
            'iou_seed_ci95': float(_ci95(ious)),
            'hd95_mean': float(np.mean(hds)),
            'hd95_seed_ci95': float(_ci95(hds)),
        })

    summary_rows.sort(key=lambda r: -r['dice_mean'])

    summary_csv = os.path.join(args.out_dir, 'ablation_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    summary_md = os.path.join(args.out_dir, 'ablation_summary.md')
    with open(summary_md, 'w') as f:
        f.write("# Ablation summary\n\n")
        f.write("Numbers are mean over seeds on the validation split recorded\n"
                "in each run's `run_metadata.json`. The ±values are 95% confidence\n"
                "intervals across seeds (normal approximation).\n\n")
        f.write("| Variant | n seeds | Dice | IoU | HD95 (px) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['variant']} | {r['n_seeds']} | "
                f"{r['dice_mean']:.3f} ± {r['dice_seed_ci95']:.3f} | "
                f"{r['iou_mean']:.3f} ± {r['iou_seed_ci95']:.3f} | "
                f"{r['hd95_mean']:.2f} ± {r['hd95_seed_ci95']:.2f} |\n"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
