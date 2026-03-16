"""
Script to extract and visualize HD_MixNet layer outputs.

Usage:
    python visualize_layers.py --path checkpoints/best_dice_model.pth --image-path test_image.jpg
"""

import torch
import argparse
import cv2
import numpy as np
from pathlib import Path

from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.inference import load_checkpoint
from Utils.transformers import get_transforms
from Utils.layer_viz import LayerOutputVisualizer, print_forward_flow


def extract_layer_outputs(model_path, image_path=None):
    """
    Extract and visualize layer outputs from HD_MixNet.

    Args:
        model_path: Path to model checkpoint
        image_path: Path to test image (if None, creates dummy image)
    """
    config = Config()

    print("\n" + "="*80)
    print("HD_MixNet LAYER OUTPUT EXTRACTION")
    print("="*80 + "\n")

    # Enable layer output storage
    config.STORE_LAYER_OUTPUTS = True

    # Load model
    print(f"Loading model from: {model_path}")
    model = HD_MixNet(num_classes=config.NUM_CLASSES, config=config).to(config.DEVICE)
    load_checkpoint(model, model_path, config.DEVICE)
    model.eval()
    print("✓ Model loaded successfully\n")

    # Prepare input
    transforms = get_transforms('test', config.IMG_SIZE)

    if image_path and Path(image_path).exists():
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms(image=image)['image']
    else:
        print("Creating dummy image (384×384×3)")
        image = torch.randn(3, config.IMG_SIZE, config.IMG_SIZE)

    # Add batch dimension
    image = image.unsqueeze(0)  # (1, 3, H, W)
    image = image.to(config.DEVICE)

    print(f"Input shape: {image.shape}\n")

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        seg_out, edge_out, aux_out = model(image)

    print("✓ Forward pass complete\n")

    # Extract layer outputs
    layer_outputs = model.layer_outputs
    print(f"Extracted {len(layer_outputs)} layer outputs\n")

    # Print forward flow diagram
    print_forward_flow(config)

    # Visualize outputs
    visualizer = LayerOutputVisualizer(output_dir='./layer_visualizations')

    print("Generating visualizations...")
    visualizer.visualize_layer_outputs(layer_outputs)

    print("Printing layer shapes...")
    visualizer.print_layer_shapes(layer_outputs)

    print("Saving statistics...")
    visualizer.save_layer_statistics(layer_outputs)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total layers tracked: {len(layer_outputs)}")
    print(f"Main output shape: {seg_out.shape}")
    print(f"Edge output shape: {edge_out.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")
    print(f"\nVisualizations saved to: layer_visualizations/")
    print(f"Statistics saved to: layer_visualizations/layer_stats.txt")
    print("\n" + "="*80 + "\n")

    return model, layer_outputs, (seg_out, edge_out, aux_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HD_MixNet layer outputs")
    parser.add_argument(
        '--path',
        type=str,
        default='./checkpoints/best_dice_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Path to test image (optional, uses dummy if not provided)'
    )
    args = parser.parse_args()

    model, layer_outputs, outputs = extract_layer_outputs(args.path, args.image_path)
