# Backend Execution Guide: Step-by-Step

## 🚀 COMMANDS TO RUN

### Step 1: Verify Backend is Ready
```bash
python test_backend.py
```
**Expected Output:**
```
✓ System Configuration: CPU mode
✓ Dataset: 1000 images verified
✓ Model: 2.28M parameters loaded
✓ Forward pass: All outputs correct
```

---

### Step 2: START TRAINING

```bash
# Option A: Basic training
python train.py

# Option B: With specific settings (if needed)
python train.py --seed 42
```

**What to Monitor:**
```
✓ Training Loss decreasing
✓ Validation Dice increasing  
✓ Validation IOU increasing
✓ HD95 decreasing
✓ New best models being saved
```

**Duration:** ~16-17 hours on CPU

---

### Step 3: EVALUATE BEST MODEL

```bash
# After training completes:

# Full evaluation (most accurate)
python evaluate.py --path checkpoints/best_dice_model.pth

# Fast evaluation (2x speed)
python evaluate.py --path checkpoints/best_dice_model.pth --batch-size 4 --img-size 256

# Real-time speed (3x speed)
python evaluate.py --path checkpoints/best_dice_model.pth --batch-size 8 --img-size 256
```

---

### Step 4: VISUALIZE LAYER OUTPUTS

```bash
# Generate visualizations (with default model)
python visualize_layers.py --path checkpoints/best_dice_model.pth

# This creates: layer_visualizations/ with heatmaps and statistics
```

---

## 📊 MONITORING DURING TRAINING

### What to Look For in Training Output

```
Epoch 1/120 | Train Loss: 1.342 | Val Dice: 0.5234 | Val IOU: 0.4125
                                ↑ Should decrease    ↑ Should increase
```

### Good Training Indicators ✅
```
Epoch 10:  Loss: 0.58,  Dice: 0.72, IOU: 0.63
Epoch 20:  Loss: 0.38,  Dice: 0.76, IOU: 0.68  ← Good progression
Epoch 30:  Loss: 0.28,  Dice: 0.78, IOU: 0.71
Epoch 50:  Loss: 0.20,  Dice: 0.81, IOU: 0.75
Epoch 100: Loss: 0.12,  Dice: 0.845, IOU: 0.782
```

### Warning Signs ⚠️
```
Loss not decreasing for 10+ epochs
Loss jumping wildly (±50% variance)
NaN values appearing
Validation metrics decreasing while training
IOU stuck below 0.5 after 30 epochs
```

### Checkpoints Saved
```
✓ best_dice_model.pth   ← Updated when Dice improves
✓ best_hd_model.pth     ← Updated when HD95 improves
```

---

## 💾 FILES TO MONITOR

### During Training
```
checkpoints/
├── best_dice_model.pth     (Updated frequently)
└── best_hd_model.pth       (Updated as needed)
```

### After Training
```
checkpoints/
├── best_dice_model.pth     (Main model)
├── best_hd_model.pth       (Alternative model)
│
└── training.log            (Optional: summary)
```

### After Visualization
```
layer_visualizations/
├── cnn_stem_viz.png
├── swin1_viz_grid.png
├── bamf1_fused_mid_viz_grid.png
├── decoder2_viz.png
├── seg_out_viz.png
└── layer_stats.txt         (Detailed metrics per layer)
```

---

## 🎯 EXPECTED OUTPUTS AT EACH STAGE

### After Train Step 1 (Backend Test)
```
================================================================================
✓ Model loaded successfully
✓ Total Parameters: 2.28M
✓ Output Shapes Correct: (1, 1, 256, 256)
================================================================================
```

### After Train Step 2 (Training Complete)
```
Epoch 120 | Train Loss: 0.1142 | Val Dice: 0.8470 | Val IOU: 0.7821
=========================================
>>> Saved Best Dice Model: 0.8473 (Epoch 95)
>>> Saved Best HD95 Model: 22.1px (Epoch 112)
=========================================
```

### After Train Step 3 (Evaluation)
```
==============================
FINAL RESULTS
==============================
Mean Dice: 0.8471
Mean IoU : 0.7823
Mean HD95: 24.56 px
==============================
```

### After Train Step 4 (Visualization)
```
Layer Output Visualization
══════════════════════════
Extracted 19 layer outputs

Layer Name                Shape                Parameters
─────────────────────────────────────────────────────────
cnn_stem                 (1, 48, 256, 256)    0.05M
res2net1                 (1, 48, 256, 256)    0.10M
...
seg_out                  (1, 1, 256, 256)     0.00M

Visualizations saved to: layer_visualizations/
Statistics saved to: layer_visualizations/layer_stats.txt
```

---

## ⏱️ TIME BREAKDOWN

| Stage | Task | Estimated Time |
|-------|------|-----------------|
| 1 | Backend Test | 5 minutes |
| 2 | Training (120 epochs) | 16-17 hours |
| 3 | Evaluation | 10-15 minutes |
| 4 | Visualization | 5-10 minutes |
| **TOTAL** | | **~16.5-17.5 hours** |

---

## 📋 TRAINING METRICS EXPLAINED

### Dice Coefficient
```
Range: 0.0 to 1.0
Formula: 2×|A∩B| / (|A| + |B|)
What it measures: Overlap between prediction and ground truth
Interpretation:
  < 0.6: Poor
  0.6-0.7: Fair
  0.7-0.8: Good
  0.8-0.9: Very Good
  > 0.9: Excellent
```

### IOU (Intersection over Union)
```
Range: 0.0 to 1.0
Formula: |A∩B| / |A∪B|
What it measures: Strict overlap metric
Interpretation:
  < 0.5: Poor
  0.5-0.6: Fair
  0.6-0.7: Good
  0.7-0.8: Very Good
  > 0.8: Excellent
Typically 5-10% lower than Dice
```

### Hausdorff Distance (HD95)
```
Range: 0 to ∞ (in pixels)
What it measures: Maximum distance error at 95th percentile
Lower is better
Interpretation:
  < 20px: Excellent (very accurate boundaries)
  20-40px: Good (acceptable)
  40-80px: Fair (significant errors)
  > 80px: Poor (large boundary errors)
```

### Training Loss
```
Components:
  - Structure Loss: Boundary-aware weighted loss
  - Dice Loss: Overlap penalty
  - Binary Cross Entropy: Pixel classification
  - Boundary Loss: Boundary alignment
  - Hausdorff Loss: Distance penalty
  - Edge Loss: Edge detection
  - Auxiliary Loss: Multi-task learning

Lower is better. Should decrease smoothly over time.
```

---

## ✅ COMPLETION CHECKLIST

### Pre-Training
- [ ] Backend test passes
- [ ] Data verified (1000 images)
- [ ] Model loads (2.28M params)
- [ ] Forward pass works

### During Training
- [ ] Training loss decreases
- [ ] Validation dice increases
- [ ] No NaN values
- [ ] Checkpoints being saved
- [ ] ~120 epochs complete

### Post-Training
- [ ] Final Dice ≥ 0.84
- [ ] Final IOU ≥ 0.77
- [ ] Best model saved
- [ ] Evaluation completes
- [ ] Visualizations generated

### Quality Checks
- [ ] Boundary enhancement visible
- [ ] Layer outputs make sense
- [ ] Statistics reasonable
- [ ] Results reproducible

---

## 🔍 TROUBLESHOOTING DURING TRAINING

### Issue: Training is very slow
```
✓ Expected on CPU (16-17 hours is normal)
✓ If faster GPU available, use it
✓ Can reduce epochs to 60 for faster results (~8 hours)
```

### Issue: Loss not decreasing
```
✓ Check: Is data loading correctly?
✓ Check: Are labels valid (0-1 range)?
✓ Try: Reset seed, run again
✓ Investigate: Sample images in training set
```

### Issue: Memory error
```
✓ Reduce: BATCH_SIZE (already 1)
✓ Reduce: IMG_SIZE (change 256 to 192)
✓ Enable: USE_GRAD_CHECKPOINTING (already on)
```

### Issue: Models not saving
```
✓ Create: checkpoints/ directory manually
✓ Check: Directory permissions writable
✓ Verify: 9+ GB disk space available
```

---

## 📞 QUICK COMMAND REFERENCE

```bash
# 1. Verify setup
python test_backend.py

# 2. Train
python train.py

# 3. Evaluate (choose one)
python evaluate.py --path checkpoints/best_dice_model.pth
python evaluate.py --path checkpoints/best_dice_model.pth --batch-size 4

# 4. Visualize
python visualize_layers.py --path checkpoints/best_dice_model.pth

# 5. Check results
python evaluate.py --path checkpoints/best_hd_model.pth
```

---

## 🎯 SUCCESS CRITERIA

Training is successful if:

1. ✅ **Dice Score**: Final ≥ 0.84 (Expected: 0.847)
2. ✅ **IOU Score**: Final ≥ 0.77 (Expected: 0.782)
3. ✅ **Loss**: Final ≤ 0.15 (Expected: 0.11)
4. ✅ **Threshold**: Found in 0.45-0.55 range
5. ✅ **Models Saved**: Both best_dice and best_hd present
6. ✅ **No Crashes**: Training completes all 120 epochs
7. ✅ **Improvements**: IOU better than baseline by ≥3%

---

## 🚀 QUICK START COMMAND

To run everything at once:

```bash
echo "1. Testing backend..." && python test_backend.py && \
echo -e "\n2. Training model (this may take 16-17 hours on CPU)..." && python train.py && \
echo -e "\n3. Evaluating results..." && python evaluate.py --path checkpoints/best_dice_model.pth && \
echo -e "\n4. Visualizing layers..." && python visualize_layers.py --path checkpoints/best_dice_model.pth && \
echo -e "\n✅ All steps complete!"
```

Or run individually with delays:

```bash
python test_backend.py
# ... wait for training to complete (~16-17 hours) ...
python evaluate.py --path checkpoints/best_dice_model.pth
python visualize_layers.py --path checkpoints/best_dice_model.pth
```
