import torch
from torch.utils.data import DataLoader
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import load_checkpoint, predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.metrics import dice_coef, iou_score, hausdorff_95
from config import Config
import argparse

def evaluate(model_path, use_tta=False, batch_size=None, img_size=None):
    config = Config()
    if batch_size is None:
        batch_size = config.INFERENCE_BATCH_SIZE
    if img_size is None:
        img_size = config.INFERENCE_IMG_SIZE

    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    checkpoint_meta = load_checkpoint(model, model_path, config.DEVICE)
    model.eval()
    threshold = float(checkpoint_meta.get('threshold', config.DEFAULT_THRESHOLD))

    test_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('test', img_size)
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Evaluating model: {model_path} on {len(test_ds)} images...")
    print(f"Using threshold={threshold:.2f}, TTA={'on' if use_tta else 'off'}, batch_size={batch_size}, img_size={img_size}")

    avg_dice = 0.0
    avg_iou = 0.0
    avg_hd95 = 0.0

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            pred_prob = predict_probabilities(model, image, use_tta=use_tta)
            pred = probabilities_to_mask(pred_prob, threshold, config=config)

            dc = dice_coef(pred, mask, from_logits=False)
            iou = iou_score(pred, mask, from_logits=False)
            hd = hausdorff_95(pred, mask, from_logits=False)

            avg_dice += dc
            avg_iou += iou
            avg_hd95 += hd

            if i % 50 == 0:
                print(f"Img {i}: Dice={dc:.4f}, IOU={iou:.4f}, HD95={hd:.2f}")

    n = len(test_loader)
    print("\n" + "="*30)
    print(f"FINAL RESULTS")
    print("="*30)
    print(f"Mean Dice: {avg_dice/n:.4f}")
    print(f"Mean IoU : {avg_iou/n:.4f}")
    print(f"Mean HD95: {avg_hd95/n:.4f} px")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoints/best_dice_model.pth', help='Path to .pth model')
    parser.add_argument('--use-tta', action='store_true', help='Enable Test Time Augmentation')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (None uses config default)')
    parser.add_argument('--img-size', type=int, default=None, help='Image size (None uses config default)')
    args = parser.parse_args()

    evaluate(args.path, use_tta=args.use_tta, batch_size=args.batch_size, img_size=args.img_size)