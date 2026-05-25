import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes

def predict_probabilities(model, images, use_tta=False):
    outputs = model(images)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    probs = [torch.sigmoid(logits)]

    if not use_tta:
        return probs[0]

    for flip_dims in ([3], [2], [2, 3]):
        aug_images = torch.flip(images, dims=flip_dims)
        aug_outputs = model(aug_images)
        aug_logits = aug_outputs[0] if isinstance(aug_outputs, (tuple, list)) else aug_outputs
        aug_probs = torch.sigmoid(aug_logits)
        probs.append(torch.flip(aug_probs, dims=flip_dims))

    return torch.stack(probs, dim=0).mean(dim=0)

def post_process_mask(
    pred_mask,
    kernel_size=5,
    keep_largest_component=True,
    min_component_area_ratio=0.001,
):
    if isinstance(pred_mask, torch.Tensor):
        device = pred_mask.device
        pred_np = pred_mask.detach().cpu().numpy().astype(np.uint8)
    else:
        device = None
        pred_np = np.asarray(pred_mask, dtype=np.uint8)

    processed = np.zeros_like(pred_np, dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for idx in range(pred_np.shape[0]):
        mask = pred_np[idx, 0]
        if not mask.any():
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = binary_fill_holes(mask > 0).astype(np.uint8)

        if keep_largest_component:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
                min_area = max(8, int(mask.size * min_component_area_ratio))
                if largest_area >= min_area:
                    mask = (labels == largest_idx).astype(np.uint8)

        processed[idx, 0] = mask

    if isinstance(pred_mask, torch.Tensor):
        return torch.from_numpy(processed).float().to(device)
    return processed.astype(np.float32)

def probabilities_to_mask(probs, threshold, config=None):
    pred_mask = (probs > threshold).float()
    if config is None or not getattr(config, 'USE_POST_PROCESSING', False):
        return pred_mask

    return post_process_mask(
        pred_mask,
        kernel_size=getattr(config, 'POST_PROCESS_KERNEL', 5),
        keep_largest_component=getattr(config, 'KEEP_LARGEST_COMPONENT', True),
        min_component_area_ratio=getattr(config, 'MIN_COMPONENT_AREA_RATIO', 0.001),
    )

def load_checkpoint(model, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "model_state_dict"
        }
        return metadata

    model.load_state_dict(checkpoint, strict=strict)
    return {}