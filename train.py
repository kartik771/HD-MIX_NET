# train.py
import os
import random
import math
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.losses import JointLoss
from Utils.metrics import dice_coef_torch, hausdorff_95, iou_score


def seed_everything(seed):
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


def split_sample_ids(sample_ids, val_split, seed):
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(sample_ids), generator=generator).tolist()

    val_size = max(1, int(len(sample_ids) * val_split))
    val_size = min(val_size, len(sample_ids) - 1)

    val_ids = [sample_ids[idx] for idx in shuffled_indices[:val_size]]
    train_ids = [sample_ids[idx] for idx in shuffled_indices[val_size:]]
    return train_ids, val_ids


def estimate_pos_weight(mask_dir, samples, min_value, max_value):
    foreground_pixels = 0.0
    total_pixels = 0.0

    for _, mask_name in samples:
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        foreground_pixels += float(mask.sum())
        total_pixels += float(mask.size)

    if foreground_pixels <= 0 or total_pixels <= 0:
        return 1.0

    background_pixels = total_pixels - foreground_pixels
    pos_weight = background_pixels / max(foreground_pixels, 1.0)
    return float(np.clip(pos_weight, min_value, max_value))


def save_checkpoint(path, model, epoch, dice, hd95, threshold):
    torch.save(
        {
            "epoch": epoch,
            "dice": float(dice),
            "hd95": float(hd95),
            "threshold": float(threshold),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def build_scheduler(optimizer, config):
    def lr_lambda(epoch_idx):
        current_epoch = epoch_idx + 1
        if current_epoch <= config.WARMUP_EPOCHS:
            return current_epoch / max(1, config.WARMUP_EPOCHS)

        progress = (current_epoch - config.WARMUP_EPOCHS) / max(
            1,
            config.NUM_EPOCHS - config.WARMUP_EPOCHS,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = config.MIN_LEARNING_RATE / config.LEARNING_RATE
        return min_scale + (1.0 - min_scale) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train():
    # 1. Setup
    config = Config()
    seed_everything(config.SEED)
    enable_speedups(config)
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # 2. Data Preparation
    base_dataset = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
    )

    if len(base_dataset) < 2:
        raise RuntimeError("Need at least 2 samples to create train/validation splits.")

    train_ids, val_ids = split_sample_ids(
        base_dataset.sample_ids,
        config.VAL_SPLIT,
        config.SEED,
    )

    train_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('train', config.IMG_SIZE),
        file_names=train_ids,
    )

    val_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('val', config.IMG_SIZE),
        file_names=val_ids,
    )

    loader_kwargs = {
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
        'persistent_workers': config.NUM_WORKERS > 0,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )

    # 3. Model & Optimization
    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    if config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    pos_weight = estimate_pos_weight(
        config.TRAIN_MASK_DIR,
        train_ds.samples,
        config.MIN_BCE_POS_WEIGHT,
        config.MAX_BCE_POS_WEIGHT,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = build_scheduler(optimizer, config)

    criterion = JointLoss(config, pos_weight=pos_weight).to(config.DEVICE)
    scaler = GradScaler(enabled=config.USE_AMP)

    best_dice = 0.0
    min_hd95 = 1e9
    last_hd95 = float('nan')
    last_threshold = config.DEFAULT_THRESHOLD

    # 4. Training Loop
    print(f"Start training HD-MixNet on {config.DEVICE} for {config.NUM_EPOCHS} epochs...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Train/Val split: {len(train_ds)} / {len(val_ds)}")
    print(
        f"Model profile: img={config.IMG_SIZE}, base_ch={config.CNN_BASE_CHANNELS}, "
        f"embed_dim={config.SWIN_EMBED_DIM}, batch={config.BATCH_SIZE}, "
        f"accum={config.ACCUMULATION_STEPS}"
    )
    print(f"Using BCE pos_weight: {pos_weight:.3f}")
    print(
        f"GPU profile: name='{config.GPU_NAME}', "
        f"vram={config.GPU_VRAM_GB:.1f} GB"
    )

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            if config.USE_CHANNELS_LAST:
                images = images.contiguous(memory_format=torch.channels_last)
            masks = masks.to(config.DEVICE, non_blocking=config.NON_BLOCKING)

            with autocast(enabled=config.USE_AMP):
                preds, edge_preds, aux_preds = model(images)

                edge_gt = F.avg_pool2d(masks, kernel_size=3, stride=1, padding=1)
                edge_gt = torch.abs(masks - edge_gt)
                edge_gt = (edge_gt > 0.1).float()

                loss_main = criterion(preds, masks)
                loss_aux = criterion(aux_preds, masks)
                loss_edge = F.binary_cross_entropy_with_logits(edge_preds, edge_gt)
                total_loss = (
                    loss_main +
                    (config.LAMBDA_AUX * loss_aux) +
                    (config.LAMBDA_EDGE * loss_edge)
                )

            loss_for_backward = total_loss / config.ACCUMULATION_STEPS
            scaler.scale(loss_for_backward).backward()

            should_step = (
                (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or
                (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})

            # Debug only once
            if epoch == 0 and batch_idx == 0:
                print("DEBUG train mask unique values:", torch.unique(masks))
                print("DEBUG images device:", images.device)
                print("DEBUG preds device:", preds.device)
                print("DEBUG aux preds shape:", aux_preds.shape)

        scheduler.step()

        # 5. Validation
        should_validate = (
            epoch == 0 or
            (epoch + 1) % config.VALIDATE_EVERY == 0 or
            (epoch + 1) == config.NUM_EPOCHS
        )
        should_compute_hd95 = (
            epoch == 0 or
            (epoch + 1) % config.HD95_EVERY == 0 or
            (epoch + 1) == config.NUM_EPOCHS
        )

        if not should_validate:
            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss / len(train_loader):.4f} | "
                f"Validation skipped"
            )
            continue

        val_dice, val_iou, val_hd95, val_threshold = validate(
            model,
            val_loader,
            config,
            compute_hd95=should_compute_hd95,
        )
        last_threshold = val_threshold
        if val_hd95 is not None:
            last_hd95 = val_hd95

        hd95_display = f"{val_hd95:.2f}" if val_hd95 is not None else "skipped"
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IOU: {val_iou:.4f} | Val HD95: {hd95_display} | "
            f"Val Thr: {val_threshold:.2f}"
        )

        # Save Best Models
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                './checkpoints/best_dice_model.pth',
                model,
                epoch + 1,
                val_dice,
                last_hd95,
                val_threshold,
            )
            print(">>> Saved Best Dice Model")

        if val_hd95 is not None and val_hd95 < min_hd95:
            min_hd95 = val_hd95
            save_checkpoint(
                './checkpoints/best_hd_model.pth',
                model,
                epoch + 1,
                val_dice,
                val_hd95,
                val_threshold,
            )
            print(">>> Saved Best HD95 Model")


def validate(model, loader, config, compute_hd95=True):
    """
    Keep threshold search on GPU and compute HD95 only once with the best threshold.
    """
    model.eval()
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            masks = masks.to(config.DEVICE, non_blocking=config.NON_BLOCKING)
            with autocast(enabled=config.USE_AMP):
                probs = predict_probabilities(model, images, use_tta=config.VAL_USE_TTA)
            all_probs.append(probs)
            all_masks.append(masks)

    best_dice = -1.0
    best_iou = -1.0
    best_threshold = config.DEFAULT_THRESHOLD

    for threshold in config.THRESHOLD_CANDIDATES:
        dice_score = 0.0
        iou_sum = 0.0

        for probs, masks in zip(all_probs, all_masks):
            pred_mask = probabilities_to_mask(probs, threshold, config=config)
            dice_score += dice_coef_torch(pred_mask, masks, threshold=0.5, from_logits=False)
            iou_sum += iou_score(pred_mask, masks, from_logits=False)

        mean_dice = dice_score / len(all_probs)
        mean_iou = iou_sum / len(all_probs)

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_iou = mean_iou
            best_threshold = threshold

    hd_score = 0.0
    if not compute_hd95:
        return best_dice, best_iou, None, best_threshold

    for probs, masks in zip(all_probs, all_masks):
        pred_mask = probabilities_to_mask(probs, best_threshold, config=config)
        hd_score += hausdorff_95(pred_mask, masks, from_logits=False)

    best_hd95 = hd_score / len(all_probs)
    return best_dice, best_iou, best_hd95, best_threshold


if __name__ == "__main__":
    train()
