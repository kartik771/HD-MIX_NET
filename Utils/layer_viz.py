import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path


class LayerOutputVisualizer:
    """Visualize and save intermediate layer outputs from HD_MixNet."""

    def __init__(self, output_dir='./layer_visualizations'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _normalize_tensor(self, tensor):
        """Normalize tensor to [0, 1] for visualization."""
        t_min = tensor.min()
        t_max = tensor.max()
        if t_max - t_min > 1e-6:
            return (tensor - t_min) / (t_max - t_min)
        return tensor

    def _save_single_channel(self, tensor, name, layer_name):
        """Save single channel as heatmap."""
        t_norm = self._normalize_tensor(tensor)
        heatmap = (t_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        path = os.path.join(self.output_dir, f"{layer_name}_{name}.png")
        cv2.imwrite(path, heatmap_colored)
        return path

    def _save_multi_channel(self, tensor, name, layer_name, max_channels=16):
        """Save multi-channel as grid of heatmaps."""
        num_channels = min(tensor.shape[0], max_channels)
        cols = min(4, num_channels)
        rows = (num_channels + cols - 1) // cols

        h, w = tensor.shape[1], tensor.shape[2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        for i in range(num_channels):
            t_norm = self._normalize_tensor(tensor[i])
            hm = (t_norm * 255).astype(np.uint8)
            hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

            row = i // cols
            col = i % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = hm_colored

        path = os.path.join(self.output_dir, f"{layer_name}_{name}_grid.png")
        cv2.imwrite(path, grid)
        return path

    def visualize_layer_outputs(self, layer_outputs, image=None):
        """
        Visualize all layer outputs from model.layer_outputs dict.

        Args:
            layer_outputs: Dict from model.layer_outputs (after forward pass)
            image: Original input image (optional, for overlay)
        """
        print(f"\n{'='*70}")
        print("LAYER OUTPUT VISUALIZATION")
        print(f"{'='*70}\n")

        for layer_name, tensor in layer_outputs.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            # Move to CPU and convert to numpy
            if tensor.is_cuda:
                tensor = tensor.cpu()

            tensor_np = tensor.squeeze(0).numpy()  # Remove batch dimension

            print(f"Layer: {layer_name:25} | Shape: {tensor_np.shape}", end="")

            # Handle different tensor shapes
            if len(tensor_np.shape) == 2:
                # Single channel (H, W)
                path = self._save_single_channel(tensor_np, "viz", layer_name)
                print(f" → Saved: {Path(path).name}")

            elif len(tensor_np.shape) == 3:
                # Multi-channel (C, H, W)
                channels = tensor_np.shape[0]
                print(f" ({channels} channels)", end="")

                if channels == 1:
                    path = self._save_single_channel(tensor_np[0], "viz", layer_name)
                    print(f" → Saved: {Path(path).name}")
                else:
                    path = self._save_multi_channel(tensor_np, "viz", layer_name)
                    print(f" → Saved: {Path(path).name}")
            else:
                print(" [Skipped: unsupported shape]")

        print(f"\nVisualizations saved to: {os.path.abspath(self.output_dir)}\n")

    def print_layer_shapes(self, layer_outputs):
        """Print summary of all layer shapes."""
        print(f"\n{'='*70}")
        print("LAYER SHAPES SUMMARY")
        print(f"{'='*70}\n")
        print(f"{'Layer Name':<30} {'Shape':<20} {'Parameters':<15}")
        print("-" * 70)

        for layer_name, tensor in sorted(layer_outputs.items()):
            if isinstance(tensor, torch.Tensor):
                shape_str = str(tuple(tensor.shape))
                num_params = tensor.numel()
                param_str = f"{num_params/1e6:.2f}M" if num_params > 1e6 else f"{num_params/1e3:.2f}K"
                print(f"{layer_name:<30} {shape_str:<20} {param_str:<15}")

        print()

    def save_layer_statistics(self, layer_outputs, filename='layer_stats.txt'):
        """Save detailed statistics of all layers."""
        stats_file = os.path.join(self.output_dir, filename)

        with open(stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HD_MixNet LAYER OUTPUT STATISTICS\n")
            f.write("="*80 + "\n\n")

            for layer_name, tensor in sorted(layer_outputs.items()):
                if not isinstance(tensor, torch.Tensor):
                    continue

                tensor_cpu = tensor.cpu() if tensor.is_cuda else tensor
                tensor_np = tensor_cpu.numpy()

                f.write(f"Layer: {layer_name}\n")
                f.write(f"  Shape: {tensor.shape}\n")
                f.write(f"  Data Type: {tensor.dtype}\n")
                f.write(f"  Memory: {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB\n")
                f.write(f"  Min: {tensor_np.min():.6f}\n")
                f.write(f"  Max: {tensor_np.max():.6f}\n")
                f.write(f"  Mean: {tensor_np.mean():.6f}\n")
                f.write(f"  Std: {tensor_np.std():.6f}\n")
                f.write(f"  Sparsity: {(tensor_np == 0).sum() / tensor_np.size * 100:.2f}%\n")
                f.write("\n")

        print(f"Statistics saved to: {stats_file}\n")


def print_forward_flow(config):
    """Print the forward pass flow for reference."""
    print(f"\n{'='*80}")
    print("HD_MixNet FORWARD PASS FLOW")
    print(f"{'='*80}\n")

    flow = """
INPUT: (B, 3, 384, 384) Image
    ↓
[CNN STREAM]                          [TRANSFORMER STREAM]
├─ cnn_stem                           ├─ patch_embed
│  (B, 48, 384, 384)                 │  (B, 96, 96, 96)
│                                     │
├─ res2net1                           ├─ swin1
│  (B, 48, 384, 384)                 │  (B, 96, 96, 96)
│  ├─ edge_detect1 ──→ (B, 48, 384) │
│                                     ├─ patch_merge
├─ res2net2                           │  (B, 192, 48, 48)
│  (B, 96, 192, 192)                 │
│  ├─ edge_detect2 ──→ (B, 96, 192) ├─ swin3
│                                     │  (B, 192, 48, 48)
├─ res2net3                           │
│  (B, 192, 96, 96)                  │
    ↓
FUSION LAYERS
├─ bamf1 (boundary-aware mix fusion)
│  CNN: (B, 96, 192, 192)
│  Trans: (B, 96, 96, 96) → upsampled
│  Edge: (B, 96, 192, 192)
│  Output: (B, 96, 192, 192)
│
├─ bamf2 (boundary-aware mix fusion)
│  CNN: (B, 192, 96, 96)
│  Trans: (B, 192, 48, 48) → upsampled
│  Output: (B, 192, 96, 96)
│
└─ context (pyramid context block)
   Output: (B, 192, 96, 96)
    ↓
DECODER (Upsampling)
├─ decoder1 (2x upsample + skip)
│  Output: (B, 96, 192, 192)
│
├─ decoder2 (2x upsample + skip)
│  Output: (B, 48, 384, 384)
│
└─ boundary_enhance
   Output: (B, 48, 384, 384)
    ↓
OUTPUT HEADS
├─ seg_out: (B, 1, 384, 384) ← Main segmentation
├─ edge_out: (B, 1, 384, 384) ← Edge map
└─ aux_out: (B, 1, 384, 384) ← Auxiliary output
"""
    print(flow)
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("Layer Output Visualizer Module")
    print("Usage:")
    print("  from Utils.layer_viz import LayerOutputVisualizer")
    print("  viz = LayerOutputVisualizer()")
    print("  viz.visualize_layer_outputs(model.layer_outputs)")
    print("  viz.print_layer_shapes(model.layer_outputs)")
    print("  viz.save_layer_statistics(model.layer_outputs)")
