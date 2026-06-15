# run_ablations.py
#
# Runs the ablation matrix the reviewer asked for. Each variant trains a
# fresh model (optionally for fewer epochs than the headline run, see
# ABLATION_EPOCHS below) and writes its checkpoints under
#     checkpoints/ablations/<variant_name>/
# along with that run's run_metadata.json and metrics_history.json.
#
# Variants (matching the reviewer's list):
#   res2net_only         : BRANCH_MODE='cnn_only',  USE_BAMF=False, USE_EDGE_SUP=False
#   swin_only            : BRANCH_MODE='swin_only', USE_BAMF=False, USE_EDGE_SUP=False
#   both_concat          : BRANCH_MODE='both', USE_BAMF=False, USE_EDGE_SUP=False
#   both_concat_edge     : BRANCH_MODE='both', USE_BAMF=False, USE_EDGE_SUP=True
#   bamf_no_edge_gate    : BRANCH_MODE='both', USE_BAMF=True,  USE_EDGE_SUP=False
#   full_no_hd_loss      : BRANCH_MODE='both', USE_BAMF=True,  USE_EDGE_SUP=True,
#                          LAMBDA_HD=0
#   full_no_ed_be        : (same as bamf_no_edge_gate but Dice+BCE only)
#   dice_bce_only        : full network but only Dice+BCE loss
#   boundary_only        : full network but boundary loss only
#   full                 : the model as published in the thesis
#
# The script does NOT swap config files on disk; it monkey-patches the
# Config class for the duration of each run, calls train.train(), and resets.
#
# Usage:
#     python run_ablations.py                  # run all variants
#     python run_ablations.py --only full,res2net_only
#     python run_ablations.py --epochs 30      # shorter runs for the ablations
#     python run_ablations.py --seeds 42 1337  # multiple seeds per variant

import argparse
import os
import shutil
import sys
import time
from copy import copy

# Make sure we can import Config from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as config_module
import train as train_module


ABLATIONS = {
    # name : dict of config overrides
    'full': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': True,
    },
    'res2net_only': {
        'BRANCH_MODE': 'cnn_only', 'USE_BAMF': False, 'USE_EDGE_SUP': False,
        'LAMBDA_EDGE': 0.0,
    },
    'swin_only': {
        'BRANCH_MODE': 'swin_only', 'USE_BAMF': False, 'USE_EDGE_SUP': False,
        'LAMBDA_EDGE': 0.0,
    },
    'both_concat': {
        'BRANCH_MODE': 'both', 'USE_BAMF': False, 'USE_EDGE_SUP': False,
        'LAMBDA_EDGE': 0.0,
    },
    'both_concat_edge': {
        'BRANCH_MODE': 'both', 'USE_BAMF': False, 'USE_EDGE_SUP': True,
    },
    'bamf_no_edge_gate': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': False,
        'LAMBDA_EDGE': 0.0,
    },
    'full_no_hd_loss': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': True,
        'LAMBDA_HD': 0.0,
    },
    'full_no_boundary_loss': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': True,
        'LAMBDA_BOUNDARY': 0.0, 'LAMBDA_HD': 0.0,
    },
    'dice_bce_only': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': True,
        'LAMBDA_STRUCT': 0.0, 'LAMBDA_BOUNDARY': 0.0, 'LAMBDA_HD': 0.0,
        'LAMBDA_EDGE': 0.0,
    },
    'boundary_only': {
        'BRANCH_MODE': 'both', 'USE_BAMF': True, 'USE_EDGE_SUP': True,
        'LAMBDA_STRUCT': 0.0, 'LAMBDA_DICE': 0.0, 'LAMBDA_BCE': 0.0,
        'LAMBDA_BOUNDARY': 1.0, 'LAMBDA_HD': 0.0,
    },
}


def patch_config(overrides, ckpt_dir, seed, epochs):
    """Mutate the Config class in place, return originals so we can restore."""
    Cfg = config_module.Config
    originals = {}
    for k, v in overrides.items():
        originals[k] = getattr(Cfg, k, None)
        setattr(Cfg, k, v)
    # Always override these too:
    originals['SEED'] = Cfg.SEED
    Cfg.SEED = seed
    originals['NUM_EPOCHS'] = Cfg.NUM_EPOCHS
    Cfg.NUM_EPOCHS = epochs
    # Redirect checkpoint paths
    originals['_RUN_METADATA_PATH'] = train_module.RUN_METADATA_PATH
    originals['_METRICS_PATH'] = train_module.METRICS_PATH
    originals['_CKPT_DIR'] = train_module.CHECKPOINT_DIR
    train_module.CHECKPOINT_DIR = ckpt_dir
    train_module.RUN_METADATA_PATH = os.path.join(ckpt_dir, 'run_metadata.json')
    train_module.METRICS_PATH = os.path.join(ckpt_dir, 'metrics_history.json')
    return originals


def restore_config(originals):
    Cfg = config_module.Config
    for k, v in originals.items():
        if k.startswith('_'):
            continue
        setattr(Cfg, k, v)
    train_module.CHECKPOINT_DIR = originals['_CKPT_DIR']
    train_module.RUN_METADATA_PATH = originals['_RUN_METADATA_PATH']
    train_module.METRICS_PATH = originals['_METRICS_PATH']


def run_one(variant_name, overrides, seed, epochs, base_dir, force):
    ckpt_dir = os.path.join(base_dir, f"{variant_name}_seed{seed}")
    if os.path.exists(ckpt_dir) and not force:
        # Skip if already done (resume-friendly).
        if os.path.exists(os.path.join(ckpt_dir, 'best_dice_model.pth')):
            print(f"[skip] {variant_name} seed={seed} -> already complete")
            return
    os.makedirs(ckpt_dir, exist_ok=True)
    originals = patch_config(overrides, ckpt_dir, seed, epochs)
    print(f"\n{'=' * 72}\n[run]  {variant_name}  seed={seed}  epochs={epochs}\n{'=' * 72}")
    t0 = time.time()
    try:
        train_module.train()
    except Exception as exc:
        print(f"[fail] {variant_name} seed={seed}: {exc!r}")
        raise
    finally:
        restore_config(originals)
    print(f"[done] {variant_name} seed={seed} in {(time.time() - t0) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='comma-separated subset of ablation names; default = all')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs per run; defaults to Config.NUM_EPOCHS')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help='seeds to run for each variant; default = Config.EVAL_SEEDS')
    parser.add_argument('--out-dir', type=str, default='./checkpoints/ablations')
    parser.add_argument('--force', action='store_true',
                        help='re-run even if a best_dice checkpoint already exists')
    args = parser.parse_args()

    Cfg = config_module.Config
    epochs = args.epochs if args.epochs is not None else Cfg.NUM_EPOCHS
    seeds = args.seeds if args.seeds is not None else list(Cfg.EVAL_SEEDS)
    only = set(args.only.split(',')) if args.only else None

    selected = {name: ov for name, ov in ABLATIONS.items() if (only is None or name in only)}
    if not selected:
        print(f"No ablation variant matched --only={args.only}.\nAvailable: {sorted(ABLATIONS)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Ablation plan: {len(selected)} variants x {len(seeds)} seeds x {epochs} epochs")

    for name, overrides in selected.items():
        for seed in seeds:
            run_one(name, overrides, seed, epochs, args.out_dir, args.force)

    print("\nAblation runs complete. Aggregate results with summarize_ablations.py.")


if __name__ == "__main__":
    main()
