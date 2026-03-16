# Layer Output Tracking & Visualization Guide

## Overview

The enhanced HD-MixNet now supports **capturing and visualizing intermediate layer outputs** during the forward pass. This is useful for:

- **Debugging**: Understanding feature flow through the network
- **Analysis**: Visualizing what each layer learns
- **Research**: Comparing CNN vs Transformer features
- **Optimization**: Identifying bottlenecks or dead layers

---

## Quick Start

### 1. Enable Layer Output Storage

**Option A: Via Config**
```python
from config import Config
config = Config()
config.STORE_LAYER_OUTPUTS = True  # Enable output tracking
```

**Option B: In config.py**
```python
# In Config class
STORE_LAYER_OUTPUTS = True  # Set to True to store intermediate activations
```

### 2. Run Visualization Script

```bash
# Using dummy image (quick test)
python visualize_layers.py --path checkpoints/best_dice_model.pth

# Using real image
python visualize_layers.py --path checkpoints/best_dice_model.pth --image-path sample.jpg
```

### 3. View Results

Outputs saved to `layer_visualizations/`:
- `*.png` - Individual layer heatmaps
- `*_grid.png` - Multi-channel feature maps
- `layer_stats.txt` - Numerical statistics

---

## All Tracked Layers

| Layer | Type | Input Shape | Output Shape | Purpose |
|-------|------|-------------|--------------|---------|
| `cnn_stem` | Conv Block | (B,3,384,384) | (B,48,384,384) | Input enhancement |
| `res2net1` | Res2Net | (B,48,384,384) | (B,48,384,384) | Stage 1 features |
| `edge_detect1` | Edge Detection | (B,48,384,384) | (B,48,384,384) | Boundary map 1 |
| `res2net2` | Res2Net | (B,48,384,384) | (B,96,192,192) | Stage 2 features |
| `edge_detect2` | Edge Detection | (B,96,192,192) | (B,96,192,192) | Boundary map 2 |
| `res2net3` | Res2Net | (B,96,192,192) | (B,192,96,96) | Stage 3 features |
| `patch_embed` | Patch Embed | (B,3,384,384) | (B,96,96,96) | ViT tokenization |
| `swin1` | Swin Stage 1 | (B,96,96,96) | (B,96,96,96) | Shallow transformer |
| `patch_merge` | Merging | (B,96,96,96) | (B,192,48,48) | Hierarchical downsample |
| `swin3` | Swin Stage 2 | (B,192,48,48) | (B,192,48,48) | Deep transformer |
| `bamf1_fused_mid` | Fusion | (B,96,192,192) | (B,96,192,192) | Mid-level fusion |
| `bamf2_before_context` | Fusion | (B,192,96,96) | (B,192,96,96) | Deep fusion (pre-context) |
| `context_block` | Context | (B,192,96,96) | (B,192,96,96) | Pyramid context |
| `decoder1` | Upsample | (B,192,96,96) | (B,96,192,192) | Stage 1 decode |
| `decoder2` | Upsample | (B,96,192,192) | (B,48,384,384) | Stage 2 decode |
| `boundary_enhance` | Enhancement | (B,48,384,384) | (B,48,384,384) | Boundary sharpening |
| `seg_out` | Output | (B,48,384,384) | (B,1,384,384) | Main segmentation |
| `edge_out` | Output | (B,48,384,384) | (B,1,384,384) | Edge prediction |
| `aux_out` | Output | (B,96,192,192) | (B,1,384,384) | Auxiliary output |

**Total Tracked Layers: 19**

---

## Usage Examples

### Example 1: Basic Visualization

```python
from config import Config
from Models.hd_mixnet import HD_MixNet
from Utils.inference import load_checkpoint
from Utils.layer_viz import LayerOutputVisualizer

# Setup
config = Config()
config.STORE_LAYER_OUTPUTS = True
model = HD_MixNet(num_classes=1, config=config).to(config.DEVICE)
load_checkpoint(model, 'checkpoints/best_dice_model.pth', config.DEVICE)
model.eval()

# Forward pass
x = torch.randn(1, 3, 384, 384).to(config.DEVICE)
with torch.no_grad():
    outputs = model(x)

# Visualize
viz = LayerOutputVisualizer()
viz.visualize_layer_outputs(model.layer_outputs)
viz.print_layer_shapes(model.layer_outputs)
viz.save_layer_statistics(model.layer_outputs)
```

### Example 2: Access Specific Layer Outputs

```python
# Get specific layer output
cnn_stem_output = model.layer_outputs['cnn_stem']  # (1, 48, 384, 384)
swin1_output = model.layer_outputs['swin1']        # (1, 96, 96, 96)
fused_features = model.layer_outputs['bamf1_fused_mid']  # (1, 96, 192, 192)

# Analyze
print(f"CNN Stem: min={cnn_stem_output.min():.4f}, max={cnn_stem_output.max():.4f}")
print(f"Swin1: mean={swin1_output.mean():.4f}, std={swin1_output.std():.4f}")
```

### Example 3: Compare CNN vs Transformer Features

```python
cnn_final = model.layer_outputs['res2net3']  # Final CNN features
trans_final = model.layer_outputs['swin3']   # Final Transformer features

print(f"CNN Feature variance: {cnn_final.var():.6f}")
print(f"Transformer Feature variance: {trans_final.var():.6f}")

# Visualize both streams
viz = LayerOutputVisualizer(output_dir='./comparison')
viz.visualize_layer_outputs({
    'cnn_stream': cnn_final,
    'transformer_stream': trans_final
})
```

### Example 4: Analyze Layer Importance

```python
# Calculate layer activations magnitude
for layer_name, tensor in model.layer_outputs.items():
    magnitude = tensor.abs().mean().item()
    print(f"{layer_name:<30} Magnitude: {magnitude:.6f}")
```

---

## Understanding Visualizations

### Heatmap Colors
- **Red**: High activation (strong features)
- **Yellow/Orange**: Medium activation
- **Blue**: Low activation (weak features)
- **Dark Blue**: Zero or near-zero activation

### Grid Format
Multi-channel layers are saved as 4-column grids:
- Each cell = one feature channel
- Brighter = stronger activation
- Darker = weaker activation

### What to Look For

| Pattern | Meaning |
|---------|---------|
| Uniform heatmap | Layer learning global features |
| Localized bright spots | Layer detecting specific patterns |
| Mostly dark/uniform | Possible dead neurons or weak layer |
| High variance | Layer capturing diverse features |

---

## Performance Impact

**Without layer tracking:**
- Memory: Baseline
- Speed: Baseline

**With layer tracking:**
- Memory: +~200MB (for storing detached tensors)
- Speed: <1% overhead (detach is cheap)

**Recommendation**: Enable only during analysis, disable for production training.

---

## Troubleshooting

### Issue: "AttributeError: 'HD_MixNet' object has no attribute 'layer_outputs'"

**Solution**: Make sure `STORE_LAYER_OUTPUTS = True` before model creation:
```python
config = Config()
config.STORE_LAYER_OUTPUTS = True
model = HD_MixNet(num_classes=1, config=config)
```

### Issue: Visualizations are all black/uniform

**Possible causes:**
1. Model not in eval mode: Add `model.eval()`
2. Layer not activated: Check if layer is being used
3. Bad initialization: Try different random seed

### Issue: CUDA out of memory

**Solution**: Reduce batch size or image size:
```python
config.INFERENCE_BATCH_SIZE = 1
config.INFERENCE_IMG_SIZE = 256
```

---

## Advanced: Custom Layer Tracking

To track additional intermediate values:

```python
# In Models/hd_mixnet.py, modify forward():
def forward(self, x):
    ...
    x_c2 = self.res2net2(x_c1)
    
    # Add custom tracking
    if self.store_layer_outputs:
        custom_feature = my_operation(x_c2)
        self.layer_outputs['my_custom_layer'] = custom_feature.detach()
    
    ...
```

---

## File Structure

```
HD-MIX_NET/
├── visualize_layers.py          ← Main script to extract outputs
├── Utils/
│   └── layer_viz.py             ← Visualization utilities
├── layer_visualizations/         ← Output directory (auto-created)
│   ├── cnn_stem_viz.png
│   ├── swin1_viz_grid.png
│   └── layer_stats.txt
└── config.py                     ← STORE_LAYER_OUTPUTS flag
```

---

## Integration with Training

To monitor layer outputs during training:

```python
# In train.py, at validation time:
model.train()
config.STORE_LAYER_OUTPUTS = True

# ... training step ...

if validation_step:
    config.STORE_LAYER_OUTPUTS = True
    with torch.no_grad():
        outputs = model(val_images)
    
    # Quick analysis
    viz = LayerOutputVisualizer(f'./checkpoints/epoch_{epoch}')
    viz.save_layer_statistics(model.layer_outputs)

config.STORE_LAYER_OUTPUTS = False  # Disable for next epoch
```

---

## References

- **Layer shapes**: See ARCHITECTURE_SUMMARY.md
- **Model config**: config.py
- **Component details**: Models/Components/
