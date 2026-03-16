# HD-MixNet Architecture & Data Flow

## 1. SYSTEM OVERVIEW

**HD-MixNet** is a Hybrid Dual-stream segmentation network for medical image segmentation (polyp detection).

**Key Concept**: Combines CNN (fast, fine details) + Vision Transformer (global context) with intelligent fusion guided by edge detection.

```
INPUT IMAGE (B, 3, H, W)
        ↓
    ┌───┴───────────────────────────────┐
    ↓                                   ↓
[CNN STREAM]                    [TRANSFORMER STREAM]
(Local Details)                 (Global Context)
    ↓                                   ↓
OUTPUT: Segmentation Map + Edge Map + Auxiliary Output
```

---

## 2. DETAILED DATA FLOW DIAGRAM

```
═════════════════════════════════════════════════════════════════════════════════════════════════════════
INPUT: Image (B, 3, 384, 384)
═════════════════════════════════════════════════════════════════════════════════════════════════════════

                                    DUAL STREAM PROCESSING
                                         ↓
        ┌────────────────────────────────┼────────────────────────────────┐
        ↓                                                                  ↓
    CNN STREAM                                                    TRANSFORMER STREAM
    ═══════════════════════════════════════════════════════════  ═══════════════════════════════════════════════════════════════
    
    1. CNN_STEM (Input Enhancement)
       ├─ Conv(3→48, k=3) + BN + ReLU
       └─ Conv(48→48, k=3) + BN + ReLU
       Output: (B, 48, 384, 384) ← x_c0
            ↓
    
    2. RES2NET_1 (Stage 1 - No Downsampling)
       ├─ Res2Net Block (4 branches)
       └─ Multi-scale feature extraction
       Output: (B, 48, 384, 384) ← x_c1
            ├─→ EDGE_DETECT_1 → (B, 48, 384, 384) [edge1]
            ↓
    
    3. RES2NET_2 (Stage 2 - 2x Downsample)
       ├─ Stride=2
       └─ Multi-scale processing
       Output: (B, 96, 192, 192) ← x_c2
            ├─→ EDGE_DETECT_2 → (B, 96, 192, 192) [edge2]
            ↓
    
    4. RES2NET_3 (Stage 3 - 4x Downsample from x_c2)
       ├─ Stride=2
       └─ Deep feature extraction
       Output: (B, 192, 96, 96) ← x_c3
    
    1. PATCH_EMBED (Input Tokenization)
       ├─ Conv(3→96, k=4, s=4) + BN + GELU
       └─ Creates 96-dim tokens
       Output: (B, 96, 96, 96) ← x_s0
            ↓
    
    2. SWIN_1 (Stage 1 - Shallow Transformer)
       ├─ Window attention (window_size=7)
       ├─ 2 transformer blocks
       ├─ Heads=3, Embed_dim=96
       └─ Global-local attention
       Output: (B, 96, 96, 96) ← x_s1
            ↓
    
    3. PATCH_MERGE (Downsampling)
       ├─ Conv(96→192, k=2, s=2) + BN + GELU
       └─ Merges neighboring patches
       Output: (B, 192, 48, 48) ← x_s2_in
            ↓
    
    4. SWIN_3 (Stage 2 - Deep Transformer)
       ├─ Window attention (window_size=7)
       ├─ 2 transformer blocks
       ├─ Heads=6, Embed_dim=192
       └─ Higher-level semantic understanding
       Output: (B, 192, 48, 48) ← x_s2

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                            FEATURE FUSION & DECODER PATH
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    Fusion Layer 1 (BAMF1 - Boundary-Aware Mix Fusion)
    ├─ CNN Input: x_c2 (B, 96, 192, 192)
    ├─ Transformer Input: x_s1 (B, 96, 96, 96) → Upsample to 192×192
    ├─ Edge Guidance: edge2 (B, 96, 192, 192)
    ├─ Gating mechanism: learns optimal mix ratio
    ├─ Edge gate boosts CNN features at boundaries
    └─ Output: fused_mid (B, 96, 192, 192) ← Mixed CNN-Transformer features
         ↓
         
    Fusion Layer 2 (BAMF2 - Boundary-Aware Mix Fusion)
    ├─ CNN Input: x_c3 (B, 192, 96, 96)
    ├─ Transformer Input: x_s2 (B, 192, 48, 48) → Upsample to 96×96
    ├─ Edge Guidance: None (deeper layer)
    ├─ Context enhancement with Pyramid Context Block
    └─ Output: fused_deep (B, 192, 96, 96) ← High-level fused features
         ↓

    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    DECODER (Progressive Upsampling)
    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    1. DECODER_1 (Stage 1 - 2x Upsample)
       ├─ Input: fused_deep (B, 192, 96, 96)
       ├─ Skip: fused_mid (B, 96, 192, 192)
       ├─ Bilinear upsample + concatenate + refine
       └─ Output: d1 (B, 96, 192, 192) ← Intermediate features
            ↓
    
    2. DECODER_2 (Stage 2 - 2x Upsample)
       ├─ Input: d1 (B, 96, 192, 192)
       ├─ Skip: x_c1 (B, 48, 384, 384)
       ├─ Bilinear upsample + concatenate + refine
       └─ Output: d2 (B, 48, 384, 384) ← Fine-grained features
            ↓
    
    3. BOUNDARY_ENHANCE_BLOCK
       ├─ Input: d2 (B, 48, 384, 384)
       ├─ Edge guidance: edge1 (B, 48, 384, 384)
       ├─ Sharpens boundaries using edge information
       └─ Output: final_feat (B, 48, 384, 384) ← Boundary-enhanced features

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
OUTPUT HEADS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    1. MAIN SEGMENTATION OUTPUT
       ├─ Conv(48→1, k=1) on final_feat
       └─ Output: seg_out (B, 1, 384, 384) ← Main prediction logits
    
    2. EDGE DETECTION OUTPUT
       ├─ Conv(48→1, k=1) on final_feat
       └─ Output: edge_out (B, 1, 384, 384) ← Edge map (auxiliary)
    
    3. AUXILIARY OUTPUT (Decoder Stage 1)
       ├─ Conv(96→1, k=1) on d1
       ├─ Upsample to seg_out size
       └─ Output: aux_out (B, 1, 384, 384) ← Auxiliary loss supervision

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
FINAL OUTPUT
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

Return: (seg_out, edge_out, aux_out)
        ├─ seg_out: Main segmentation logits (B, 1, 384, 384)
        ├─ edge_out: Edge prediction (B, 1, 384, 384)
        └─ aux_out: Auxiliary prediction from mid-level features (B, 1, 384, 384)
```

---

## 3. KEY ARCHITECTURAL COMPONENTS

### **A. CNN Stream (Res2Net)**
| Stage | Module | Input Shape | Output Shape | Purpose |
|-------|--------|-------------|--------------|---------|
| 0 | CNN_STEM | (B,3,384,384) | (B,48,384,384) | Input normalization |
| 1 | Res2Net_1 | (B,48,384,384) | (B,48,384,384) | Fine local features |
| 2 | Res2Net_2 | (B,48,384,384) | (B,96,192,192) | Mid-level features |
| 3 | Res2Net_3 | (B,96,192,192) | (B,192,96,96) | Deep global features |

**Why Res2Net?** Multi-scale branch design captures features at different receptive field sizes.

### **B. Transformer Stream (Swin)**
| Stage | Module | Input Shape | Output Shape | Purpose |
|-------|--------|-------------|--------------|---------|
| Embed | Patch_Embed | (B,3,384,384) | (B,96,96,96) | Image→Tokens (4×4 patches) |
| 1 | Swin_1 | (B,96,96,96) | (B,96,96,96) | Shallow global context |
| Merge | Patch_Merge | (B,96,96,96) | (B,192,48,48) | Hierarchical downsampling |
| 2 | Swin_3 | (B,192,48,48) | (B,192,48,48) | Deep global context |

**Why Swin?** Window-based attention is efficient and captures both local & global patterns.

### **C. Boundary-Aware Mix Fusion (BAMF)**
**Input**: CNN features + Transformer features + Optional Edge guidance
**Process**:
1. Project both streams to same dimension
2. Mix gate: learns blend ratio σ = sigmoid(concat[CNN, Trans])
3. Fused = σ⊙CNN + (1-σ)⊙Trans  (element-wise blend)
4. Edge gate: If edges present, boost CNN at edges (sharp details)
5. Refinement: 2×Conv + Squeeze-Excite + residual connection

**Why?** CNN excels at fine details; Transformer at global context. Fusion learns optimal blend.

---

## 4. LOSS FUNCTIONS (Training)

```python
Total Loss = λ_struct × L_struct 
           + λ_dice × L_dice
           + λ_bce × L_bce
           + λ_boundary × L_boundary
           + λ_hd × L_hausdorff
           + λ_edge × L_edge
           + λ_aux × L_aux
```

| Loss | Weight | Purpose |
|------|--------|---------|
| Structure | 1.0 | Boundary-aware weighted BCE + IoU |
| Dice | 0.4 | Overlap metric (1 - 2∩/∪) |
| BCE | 0.2 | Pixel-level classification |
| Boundary | 0.40 | Boundary alignment (was 0.25) |
| Hausdorff | 0.08 | Boundary distance (NEW) |
| Edge | 0.15 | Edge map supervision |
| Auxiliary | 0.35 | Mid-level feature supervision |

---

## 5. DATA SHAPES AT EACH LAYER

```
Layer                           Shape                   Channels
─────────────────────────────────────────────────────────────────
INPUT                          (B, 3, 384, 384)        3
CNN_STEM                       (B, 48, 384, 384)       48 ← x_c0
RES2NET_1                      (B, 48, 384, 384)       48 ← x_c1
  ├─ EDGE_DETECT_1             (B, 48, 384, 384)       48 ← edge1
RES2NET_2                      (B, 96, 192, 192)       96 ← x_c2
  ├─ EDGE_DETECT_2             (B, 96, 192, 192)       96 ← edge2
RES2NET_3                      (B, 192, 96, 96)        192 ← x_c3

PATCH_EMBED                    (B, 96, 96, 96)         96 ← x_s0
SWIN_1                         (B, 96, 96, 96)         96 ← x_s1
PATCH_MERGE                    (B, 192, 48, 48)        192 ← x_s2_in
SWIN_3                         (B, 192, 48, 48)        192 ← x_s2

BAMF1 (fused_mid)              (B, 96, 192, 192)       96
BAMF2 + Context (fused_deep)   (B, 192, 96, 96)        192

DECODER_1                      (B, 96, 192, 192)       96 ← d1
DECODER_2                      (B, 48, 384, 384)       48 ← d2
BOUNDARY_ENHANCE               (B, 48, 384, 384)       48 ← final_feat

SEG_OUT                        (B, 1, 384, 384)        1
EDGE_OUT                       (B, 1, 384, 384)        1
AUX_OUT                        (B, 1, 384, 384)        1
```

---

## 6. INFERENCE PIPELINE

```
Model Input (384×384)
    ↓
Forward Pass (with layer tracking)
    ├─→ Store CNN features (x_c0, x_c1, x_c2, x_c3)
    ├─→ Store Transformer features (x_s0, x_s1, x_s2)
    ├─→ Store Edge maps (edge1, edge2)
    ├─→ Store Fused features (fused_mid, fused_deep)
    ├─→ Store Decoder outputs (d1, d2)
    └─→ Return outputs + intermediate activations
    ↓
Post-Processing
    ├─ Sigmoid: logits → probabilities [0,1]
    ├─ Threshold: probabilities → binary mask
    ├─ Morphological ops: clean noisy pixels
    └─ Keep largest component
    ↓
Metrics Computation
    ├─ IOU = |∩| / |∪|
    ├─ Dice = 2|∩| / (|pred| + |gt|)
    └─ Hausdorff95 = 95th percentile of distances
    ↓
Output: Segmentation mask + metrics
```

---

## 7. PARAMETER COUNTS

```
Component             Parameters    Trainable
──────────────────────────────────────────────
CNN Stream:
  CNN_STEM            ~3K           Yes
  Res2Net blocks      ~150K         Yes
  Edge Detection      ~50K          Yes
Subtotal              ~203K

Transformer Stream:
  Patch Embedding     ~80K          Yes
  Swin_1              ~200K         Yes
  Swin_3              ~450K         Yes
Subtotal              ~730K

Fusion & Decoder:
  BAMF blocks         ~100K         Yes
  Context Block       ~50K          Yes
  Decoder blocks      ~80K          Yes
  Output heads        ~10K          Yes
Subtotal              ~240K

──────────────────────────────────────────────
TOTAL MODEL SIZE      ~1.17M parameters
```

---

## 8. COMPUTATIONAL FLOW (SUMMARY)

```
┌──────────────────────────────────────────────────────────────────┐
│  1. INPUT PROCESSING                                             │
│  └─ Pad input to multiple of 56 (window_size × 8)                │
│                                                                   │
│  2. PARALLEL ENCODING (CNN + Vision Transformer)                 │
│  ├─ CNN: 3 downsampling stages (×1, ×2, ×4)                      │
│  └─ ViT: Patch embedding, 2-stage transformer                    │
│                                                                   │
│  3. BOUNDARY-AWARE FUSION                                         │
│  ├─ Mix CNN local details + Transformer global context           │
│  └─ Edge detection gates the fusion (sharper at boundaries)      │
│                                                                   │
│  4. HIERARCHICAL DECODER                                          │
│  ├─ Progressive upsampling with skip connections                 │
│  ├─ Boundary enhancement at final layer                          │
│  └─ 3 output heads (main, edge, auxiliary)                       │
│                                                                   │
│  5. TRAINING OPTIMIZATION                                         │
│  ├─ Multiple loss terms (structure, dice, boundary, hausdorff)  │
│  ├─ Gradient checkpointing (memory efficiency)                   │
│  └─ AMP (Automatic Mixed Precision)                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. KEY OPTIMIZATIONS IN CODE

| Optimization | Impact | Where |
|--------------|--------|-------|
| **Gradient Checkpointing** | -40% GPU RAM | `_forward_with_checkpoint()` |
| **AMP (Automatic Mixed Precision)** | 1.5-2x faster | `autocast()` in train loop |
| **Channels Last Format** | +15% inference speed | `to(channels_last)` |
| **TF32 Precision** | 3-4x faster matmul | `allow_tf32=True` |
| **Input Padding** | No information loss | `_pad_input()` |

