# test.py
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import load_checkpoint, predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.metrics import dice_coef, hausdorff_95


def save_results(image, mask, pred_binary, pred_prob, save_dir, image_name):
    """
    Visualization:

    [ white label strip ]
    [ Original | Ground Truth | Predicted Probabilities | Overlay ]

    - Each panel has a black border.
    - Text only in the white strip (images themselves are untouched).
    - Overlay: GT = green, Pred (binary) = red.
    """

    # ---------- 1. Base image (denormalize, RGB -> BGR) ----------
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ---------- 2. Prepare mask & predictions ----------
    mask = np.squeeze(mask)
    pred_binary = np.squeeze(pred_binary)
    pred_prob = np.squeeze(pred_prob)

    # GT: 0/1 or 0/255 -> 0/255
    if mask.max() <= 1.0:
        mask_vis = (mask * 255).astype(np.uint8)
    else:
        mask_vis = mask.astype(np.uint8)

    # Binary pred: 0/1 -> 0/255
    if pred_binary.max() <= 1.0:
        pred_bin_vis = (pred_binary * 255).astype(np.uint8)
    else:
        pred_bin_vis = pred_binary.astype(np.uint8)

    # Probabilities: [0,1] -> [0,255]
    pred_prob_vis = np.clip(pred_prob, 0.0, 1.0)
    pred_prob_vis = (pred_prob_vis * 255).astype(np.uint8)

    # ---------- 3. Panels ----------
    # GT panel
    mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # Predicted probability heatmap panel
    pred_prob_heatmap = cv2.applyColorMap(pred_prob_vis, cv2.COLORMAP_JET)

    # Overlay panel: GT=green, Pred=red
    overlay = image_bgr.copy()

    gt_layer = np.zeros_like(image_bgr)
    gt_layer[mask_vis > 127] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, gt_layer, 0.4, 0)

    pred_layer = np.zeros_like(image_bgr)
    pred_layer[pred_bin_vis > 127] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, pred_layer, 0.4, 0)

    contours_gt, _ = cv2.findContours(mask_vis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_bin_vis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, contours_pred, -1, (0, 0, 255), 2)

    # ---------- 4. Borders ----------
    panels = [image_bgr, mask_rgb, pred_prob_heatmap, overlay]
    bordered_panels = []
    border_size = 3
    border_color = (0, 0, 0)

    for p in panels:
        p_bordered = cv2.copyMakeBorder(
            p, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=border_color
        )
        bordered_panels.append(p_bordered)

    combined = np.hstack(bordered_panels)

    # ---------- 5. White label strip ----------
    labels = [
        "Original Image",
        "Ground Truth Mask",
        "Predicted Probabilities ",
        "Overlay (GT: Green, Pred: Red)",
    ]

    num_panels = len(labels)
    h, w, _ = combined.shape
    label_strip_h = 50

    label_strip = np.full((label_strip_h, w, 3), 255, dtype=np.uint8)
    panel_w = w // num_panels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (0, 0, 0)

    for idx, label in enumerate(labels):
        start_x = idx * panel_w
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = start_x + (panel_w - text_w) // 2
        text_y = (label_strip_h + text_h) // 2

        cv2.putText(label_strip, label, (text_x, text_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

    final_image = np.vstack([label_strip, combined])
    cv2.imwrite(os.path.join(save_dir, image_name), final_image)


def test(model_path, save_visuals=True):
    config = Config()

    save_dir = './results/visuals'
    if save_visuals and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading model from {model_path}...")
    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    checkpoint_meta = load_checkpoint(model, model_path, config.DEVICE)
    model.eval()

    threshold = float(checkpoint_meta.get('threshold', config.DEFAULT_THRESHOLD))

    test_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR,     # change to TEST_* if you have them
        mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('test', config.IMG_SIZE)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    print(f"Starting Inference on {len(test_ds)} images...")
    print(f"Using threshold={threshold:.2f}, TTA={'on' if config.USE_TTA else 'off'}")

    metrics = {'dice': [], 'hd95': []}

    with torch.no_grad():
        for i, (image_tensor, mask_tensor) in enumerate(tqdm(test_loader)):
            image_tensor = image_tensor.to(config.DEVICE)
            mask_tensor = mask_tensor.to(config.DEVICE)

            pred_prob = predict_probabilities(model, image_tensor, use_tta=config.USE_TTA)
            pred_mask = probabilities_to_mask(pred_prob, threshold, config=config)

            if i == 0:
                print("DEBUG (first test sample):")
                print("  pred_prob min/max:",
                      pred_prob.min().item(), pred_prob.max().item())
                print("  unique in pred_mask:", torch.unique(pred_mask))
                print("  unique in GT mask:  ", torch.unique(mask_tensor))

            d = dice_coef(pred_mask, mask_tensor, from_logits=False)
            h = hausdorff_95(pred_mask, mask_tensor, from_logits=False)
            metrics['dice'].append(d)
            metrics['hd95'].append(h)

            if save_visuals:
                img_np = (
                    image_tensor[0]
                    .detach().cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )

                mask_np = (
                    mask_tensor[0]
                    .detach().cpu()
                    .squeeze()
                    .numpy()
                )

                pred_bin_np = (
                    pred_mask[0]
                    .detach().cpu()
                    .squeeze()
                    .numpy()
                )

                pred_prob_np = (
                    pred_prob[0]
                    .detach().cpu()
                    .squeeze()
                    .numpy()
                )

                image_name = f"result_{i:04d}_dice_{float(d):.3f}.png"
                save_results(img_np, mask_np, pred_bin_np, pred_prob_np,
                             save_dir, image_name)

    mean_dice = float(np.mean(metrics['dice'])) if metrics['dice'] else 0.0
    mean_hd95 = float(np.mean(metrics['hd95'])) if metrics['hd95'] else 0.0

    print("\n" + "=" * 40)
    print("TESTING COMPLETE")
    print("=" * 40)
    print(f"Mean Dice Coefficient : {mean_dice:.4f}")
    print(f"Mean HD95             : {mean_hd95:.4f} px")
    print(f"Visual results saved to: {save_dir}")
    print("=" * 40)


if __name__ == "__main__":
    MODEL_PATH = './checkpoints/best_dice_model.pth'
    if os.path.exists(MODEL_PATH):
        test(MODEL_PATH)
    else:
        print(f"Checkpoint not found at {MODEL_PATH}")
