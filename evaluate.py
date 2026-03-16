# evaluate.py
import torch
from torch.utils.data import DataLoader
from Models.hd_mixnet import HD_MixNet
from Utils.dataset import KvasirDataset
from Utils.inference import load_checkpoint, predict_probabilities, probabilities_to_mask
from Utils.transformers import get_transforms
from Utils.metrics import dice_coef, iou_score, hausdorff_95
from config import Config
import argparse

def evaluate(model_path):
    config = Config()
    

    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    checkpoint_meta = load_checkpoint(model, model_path, config.DEVICE)
    model.eval()
    threshold = float(checkpoint_meta.get('threshold', config.DEFAULT_THRESHOLD))

    test_ds = KvasirDataset(
        img_dir=config.TRAIN_IMG_DIR, # Or config.TEST_IMG_DIR if separated
        mask_dir=config.TRAIN_MASK_DIR,
        transforms=get_transforms('test', config.IMG_SIZE)
    )
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Evaluating model: {model_path} on {len(test_ds)} images...")
    print(f"Using threshold={threshold:.2f}, TTA={'on' if config.USE_TTA else 'off'}")
    
    avg_dice = 0.0
    avg_iou = 0.0
    avg_hd95 = 0.0
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            
            # Inference
            pred_prob = predict_probabilities(model, image, use_tta=config.USE_TTA)
            pred = probabilities_to_mask(pred_prob, threshold, config=config)
            
            # Metrics
            dc = dice_coef(pred, mask, from_logits=False)
            iou = iou_score(pred, mask, from_logits=False)
            hd = hausdorff_95(pred, mask, from_logits=False)
            
            avg_dice += dc
            avg_iou += iou
            avg_hd95 += hd
            
            if i % 50 == 0:
                print(f"Img {i}: Dice={dc:.4f}, HD95={hd:.2f}")

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
    parser.add_argument('--path', type=str, default='./checkpoints/best_dice_model.pth', help='Path to.pth model')
    args = parser.parse_args()
    
    evaluate(args.path)

