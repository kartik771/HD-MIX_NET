# train.py
#
# Changes:
#   - writes checkpoints/run_metadata.json at the start of training with the
#     full hyperparameter set, the resolved train/val sample IDs and the
#     random seed (reproducibility issue raised by the reviewer).
#   - supports config.THRESHOLD_SELECTION = "dice" | "hd95" | "composite",
#     so that the threshold can be selected to be consistent with the
#     thesis's boundary-quality emphasis (the previous behaviour, "dice",
#     was internally inconsistent with the boundary-aware claim).
#   - logs train_loss / val_dice / val_iou / val_hd95 / val_threshold /
#     learning_rate per epoch to checkpoints/metrics_history.json after every
#     validated epoch (atomic write).

import os
import json
import math
import random
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.losses import JointLoss
from Utils.metrics import dice_coef_torch, hausdorff_95, iou_score


CHECKPOINT_DIR = './checkpoints'
RUN_METADATA_PATH = os.path.join(CHECKPOINT_DIR, 'run_metadata.json')
METRICS_PATH = os.path.join(CHECKPOINT_DIR, 'metrics_history.json')


# ---------------------------------------------------------------------------
# Reproducibility helpers

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def enable_speedups(config):
    torch.set_float32_matmul_precision("high")
    if config.DEVICE.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = config.USE_TF32
        torch.backends.cudnn.allow_tf32 = config.USE_TF32


def split_sample_ids(sample_ids, val_split, seed, test_split=0.0):
    """Returns (train_ids, val_ids, test_ids) with fixed-seed shuffling."""
    generator = torch.Generator().manual_seed(seed)
    n = len(sample_ids)
    shuffled = torch.randperm(n, generator=generator).tolist()

    n_val = max(1, int(n * val_split))
    n_test = int(n * test_split)
    n_val = min(n_val, max(1, n - n_test - 1))

    test_idx = shuffled[:n_test]
    val_idx = shuffled[n_test:n_test + n_val]
    train_idx = shuffled[n_test + n_val:]

    test_ids = [sample_ids[i] for i in test_idx]
    val_ids = [sample_ids[i] for i in val_idx]
    train_ids = [sample_ids[i] for i in train_idx]
    return train_ids, val_ids, test_ids


def estimate_pos_weight(mask_dir, samples, min_value, max_value):
    fg = 0.0
    total = 0.0
    for _, mask_name in samples:
        mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        fg += float(mask.sum())
        total += float(mask.size)
    if fg <= 0 or total <= 0:
        return 1.0
    pw = (total - fg) / max(fg, 1.0)
    return float(np.clip(pw, min_value, max_value))


def build_scheduler(optimizer, config):
    def lr_lambda(epoch_idx):
        e = epoch_idx + 1
        if e <= config.WARMUP_EPOCHS:
            return e / max(1, config.WARMUP_EPOCHS)
        progress = (e - config.WARMUP_EPOCHS) / max(1, config.NUM_EPOCHS - config.WARMUP_EPOCHS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = config.MIN_LEARNING_RATE / config.LEARNING_RATE
        return min_scale + (1.0 - min_scale) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path, model, epoch, dice, hd95, threshold, config):
    torch.save(
        {
            "epoch": epoch,
            "dice": float(dice),
            "hd95": float(hd95) if hd95 is not None else None,
            "threshold": float(threshold),
            "model_state_dict": model.state_dict(),
            "branch_mode": getattr(config, 'BRANCH_MODE', 'both'),
            "use_bamf": bool(getattr(config, 'USE_BAMF', True)),
            "use_edge_sup": bool(getattr(config, 'USE_EDGE_SUP', True)),
        },
        path,
    )


def _atomic_dump_json(obj, path):
    """Write JSON atomically to avoid corrupting on crash mid-write."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp_', suffix='.json', dir=os.path.dirname(path) or '.')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _collect_hparams(config):
    keys = [
        'SEED', 'IMG_SIZE', 'BATCH_SIZE', 'ACCUMULATION_STEPS',
        'LEARNING_RATE', 'MIN_LEARNING_RATE', 'NUM_EPOCHS', 'WARMUP_EPOCHS',
        'WEIGHT_DECAY', 'GRAD_CLIP_NORM',
        'VAL_SPLIT', 'TEST_SPLIT',
        'CNN_BASE_CHANNELS', 'SWIN_EMBED_DIM', 'SWIN_WINDOW_SIZE',
        'SWIN_HEADS_STAGE1', 'SWIN_HEADS_STAGE2', 'SWIN_STAGE_DEPTHS',
        'SWIN_MLP_RATIO', 'SWIN_DROP_PATH', 'RES2NET_SCALE',
        'BRANCH_MODE', 'USE_BAMF', 'USE_EDGE_SUP',
        'LAMBDA_STRUCT', 'LAMBDA_DICE', 'LAMBDA_BCE',
        'LAMBDA_BOUNDARY', 'LAMBDA_HD', 'LAMBDA_EDGE', 'LAMBDA_AUX',
        'STRUCTURE_POOL_KERNEL', 'BOUNDARY_LOSS_KERNEL',
        'HD_ALPHA', 'HD_BINARIZE_THRESHOLD', 'HD_NORMALIZE',
        'DEFAULT_THRESHOLD', 'THRESHOLD_CANDIDATES', 'THRESHOLD_SELECTION',
        'THRESHOLD_COMPOSITE_LAMBDA', 'THRESHOLD_COMPOSITE_HD_NORM',
        'USE_TTA', 'VAL_USE_TTA', 'USE_POST_PROCESSING',
        'POST_PROCESS_KERNEL', 'KEEP_LARGEST_COMPONENT', 'MIN_COMPONENT_AREA_RATIO',
    ]
    out = {}
    for k in keys:
        if hasattr(config, k):
            v = getattr(config, k)
            if isinstance(v, tuple):
                v = list(v)
            out[k] = v
    return out


def write_run_metadata(config, train_ids, val_ids, test_ids):
    meta = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'device': str(config.DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
        'train_ids': sorted(train_ids),
        'val_ids': sorted(val_ids),
        'test_ids': sorted(test_ids),
        'hparams': _collect_hparams(config),
    }
    _atomic_dump_json(meta, RUN_METADATA_PATH)


# ---------------------------------------------------------------------------
# Validation with configurable threshold selection

def validate(model, loader, config, compute_hd95=True):
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            masks = masks.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            with autocast(enabled=config.USE_AMP):
                probs = predict_probabilities(model, images, use_tta=config.VAL_USE_TTA)
            all_probs.append(probs)
            all_masks.append(masks)

    selection_mode = getattr(config, 'THRESHOLD_SELECTION', 'dice').lower()
    composite_lambda = float(getattr(config, 'THRESHOLD_COMPOSITE_LAMBDA', 0.05))
    hd_norm = float(getattr(config, 'THRESHOLD_COMPOSITE_HD_NORM', 50.0))

    # If we select on HD or composite we need HD95 at every candidate; that is
    # expensive but is the *point* of the option. For pure-Dice selection
    # we keep the original cheap path.
    candidates = list(config.THRESHOLD_CANDIDATES)

    per_threshold = []
    for thr in candidates:
        dice_sum = 0.0
        iou_sum = 0.0
        hd_sum = 0.0
        hd_valid = 0
        for probs, masks in zip(all_probs, all_masks):
            pred_mask = probabilities_to_mask(probs, thr, config=config)
            dice_sum += dice_coef_torch(pred_mask, masks, threshold=0.5, from_logits=False)
            iou_sum += iou_score(pred_mask, masks, from_logits=False)
            if selection_mode != 'dice' and compute_hd95:
                hd = hausdorff_95(pred_mask, masks, from_logits=False)
                hd_sum += hd
                hd_valid += 1
        n = len(all_probs)
        mean_dice = dice_sum / n
        mean_iou = iou_sum / n
        mean_hd = (hd_sum / hd_valid) if hd_valid > 0 else None
        per_threshold.append({'threshold': thr, 'dice': mean_dice, 'iou': mean_iou, 'hd95': mean_hd})

    # Choose best threshold
    if selection_mode == 'dice' or not compute_hd95:
        best = max(per_threshold, key=lambda r: r['dice'])
    elif selection_mode == 'hd95':
        # smaller is better; tie-break with Dice
        best = min(per_threshold, key=lambda r: (r['hd95'] if r['hd95'] is not None else float('inf'),
                                                 -r['dice']))
    elif selection_mode == 'composite':
        def score(r):
            hd = r['hd95'] if r['hd95'] is not None else hd_norm
            return r['dice'] - composite_lambda * (hd / hd_norm)
        best = max(per_threshold, key=score)
    else:
        raise ValueError(f"Unknown THRESHOLD_SELECTION: {selection_mode}")

    # Always ensure we report HD95 at the chosen threshold (recompute if dice-mode skipped it).
    if compute_hd95 and best['hd95'] is None:
        hd_sum = 0.0
        n = 0
        for probs, masks in zip(all_probs, all_masks):
            pred_mask = probabilities_to_mask(probs, best['threshold'], config=config)
            hd_sum += hausdorff_95(pred_mask, masks, from_logits=False)
            n += 1
        best['hd95'] = hd_sum / max(n, 1)

    return best['dice'], best['iou'], best['hd95'], best['threshold']


# ---------------------------------------------------------------------------
# Training loop

def train():
    config = Config()
    seed_everything(config.SEED)
    enable_speedups(config)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    base_dataset = KvasirDataset(img_dir=config.TRAIN_IMG_DIR, mask_dir=config.TRAIN_MASK_DIR)
    if len(base_dataset) < 2:
        raise RuntimeError("Need at least 2 samples to create train/validation splits.")

    train_ids, val_ids, test_ids = split_sample_ids(
        base_dataset.sample_ids,
        config.VAL_SPLIT,
        config.SEED,
        test_split=getattr(config, 'TEST_SPLIT', 0.0),
    )
    write_run_metadata(config, train_ids, val_ids, test_ids)

    train_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR, mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('train', config.IMG_SIZE), file_names=train_ids,
    )
    val_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR, mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('val', config.IMG_SIZE), file_names=val_ids,
    )

    loader_kwargs = {
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
        'persistent_workers': config.NUM_WORKERS > 0,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)

    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    if config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    pos_weight = estimate_pos_weight(
        config.TRAIN_MASK_DIR, train_ds.samples,
        config.MIN_BCE_POS_WEIGHT, config.MAX_BCE_POS_WEIGHT,
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, config)
    criterion = JointLoss(config, pos_weight=pos_weight).to(config.DEVICE)
    scaler = GradScaler(enabled=config.USE_AMP)

    best_dice, min_hd95 = 0.0, float('inf')
    last_hd95 = float('nan')

    metrics_history = {
        'epochs': [], 'train_loss': [], 'val_dice': [], 'val_iou': [],
        'val_hd95': [], 'val_threshold': [], 'learning_rate': [],
    }

    print(f"HD-MixNet training on {config.DEVICE} for {config.NUM_EPOCHS} epochs")
    print(f"  CUDA available: {torch.cuda.is_available()}  |  Using CUDA: {config.DEVICE.type == 'cuda'}")
    print(f"  Branch mode: {config.BRANCH_MODE}  |  BAMF: {config.USE_BAMF}  |  Edge sup: {config.USE_EDGE_SUP}")
    print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ids)}  |  pos_weight: {pos_weight:.3f}")
    print(f"  Threshold selection: {getattr(config, 'THRESHOLD_SELECTION', 'dice')}")
    print(f"  Run metadata: {RUN_METADATA_PATH}")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            if config.USE_CHANNELS_LAST:
                images = images.contiguous(memory_format=torch.channels_last)
            masks = masks.to(config.DEVICE, non_blocking=config.NON_BLOCKING)

            with autocast(enabled=config.USE_AMP):
                preds, edge_preds, aux_preds = model(images)

                loss_main = criterion(preds, masks)
                loss_aux = criterion(aux_preds, masks)

                if config.USE_EDGE_SUP and config.LAMBDA_EDGE > 0:
                    edge_gt = F.avg_pool2d(masks, kernel_size=3, stride=1, padding=1)
                    edge_gt = (torch.abs(masks - edge_gt) > 0.1).float()
                    loss_edge = F.binary_cross_entropy_with_logits(edge_preds, edge_gt)
                else:
                    loss_edge = preds.new_zeros(())

                total_loss = loss_main + config.LAMBDA_AUX * loss_aux + config.LAMBDA_EDGE * loss_edge

            scaler.scale(total_loss / config.ACCUMULATION_STEPS).backward()

            if ((batch_idx + 1) % config.ACCUMULATION_STEPS == 0
                    or (batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += total_loss.item()
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

        scheduler.step()

        should_validate = (epoch == 0 or (epoch + 1) % config.VALIDATE_EVERY == 0
                           or (epoch + 1) == config.NUM_EPOCHS)
        should_hd95 = (epoch == 0 or (epoch + 1) % config.HD95_EVERY == 0
                       or (epoch + 1) == config.NUM_EPOCHS)

        if not should_validate:
            print(f"Epoch {epoch + 1} | loss {running_loss / len(train_loader):.4f} | val skipped")
            continue

        val_dice, val_iou, val_hd95, val_thr = validate(model, val_loader, config, compute_hd95=should_hd95)
        if val_hd95 is not None:
            last_hd95 = val_hd95

        hd_str = f"{val_hd95:.2f}" if val_hd95 is not None else "skipped"
        print(
            f"Epoch {epoch + 1} | loss {running_loss / len(train_loader):.4f} | "
            f"dice {val_dice:.4f} | iou {val_iou:.4f} | hd95 {hd_str} | thr {val_thr:.2f}"
        )

        metrics_history['epochs'].append(epoch + 1)
        metrics_history['train_loss'].append(float(running_loss / len(train_loader)))
        metrics_history['val_dice'].append(float(val_dice))
        metrics_history['val_iou'].append(float(val_iou))
        metrics_history['val_hd95'].append(float(val_hd95) if val_hd95 is not None else None)
        metrics_history['val_threshold'].append(float(val_thr))
        metrics_history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        _atomic_dump_json(metrics_history, METRICS_PATH)

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(os.path.join(CHECKPOINT_DIR, 'best_dice_model.pth'),
                            model, epoch + 1, val_dice, last_hd95, val_thr, config)
            print(">>> saved best-Dice checkpoint")

        if val_hd95 is not None and val_hd95 < min_hd95:
            min_hd95 = val_hd95
            save_checkpoint(os.path.join(CHECKPOINT_DIR, 'best_hd_model.pth'),
                            model, epoch + 1, val_dice, val_hd95, val_thr, config)
            print(">>> saved best-HD95 checkpoint")


if __name__ == "__main__":
    train()
