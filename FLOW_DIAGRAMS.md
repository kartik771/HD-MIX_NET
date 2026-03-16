# HD-MixNet: Visual Architecture & Flow Diagrams

## 1. HIGH-LEVEL ARCHITECTURE

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          HD-MIXNET: DUAL STREAM                               ║
║                    CNN + Vision Transformer Hybrid                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  INPUT IMAGE: (Batch, 3, 384, 384)                                          ║
║         │                                                                     ║
║         ├─────────────────────┬────────────────────────┬────────────────────┤
║         ↓                     ↓                        ↓                      ║
║    ┌─────────┐          ┌──────────┐            ┌──────────┐               ║
║    │   CNN   │          │Transformer               │Aux Input│              ║
║    │ STREAM  │          │  STREAM   │            │Processing              ║
║    │(Res2Net)│          │ (Swin)    │            │          │              ║
║    └────┬────┘          └─────┬─────┘            └────┬─────┘              ║
║         │                     │                       │                     ║
║         ├─ CNN_STEM           ├─ Patch Embed         │                     ║
║         ├─ Res2Net_1          ├─ Swin_1 (2 blocks)   │                     ║
║         ├─ Res2Net_2          ├─ Patch Merge         │                     ║
║         └─ Res2Net_3          └─ Swin_3 (2 blocks)   │                     ║
║         │                     │                       │                     ║
║         └──────────┬──────────┘                       │                     ║
║                    ↓                                  │                     ║
║         ┌────────────────────────┐                   │                     ║
║         │  BOUNDARY-AWARE FUSION │                   │                     ║
║         │ (BAMF: 2 levels)       │                   │                     ║
║         │ - Mix CNN + Trans      │                   │                     ║
║         │ - Edge-guided blending │                   │                     ║
║         └────────────┬───────────┘                   │                     ║
║                      ↓                                │                     ║
║         ┌────────────────────────┐                   │                     ║
║         │  PYRAMID CONTEXT BLOCK │                   │                     ║
║         │  (Multi-scale context) │                   │                     ║
║         └────────────┬───────────┘                   │                     ║
║                      ↓                                │                     ║
║         ┌────────────────────────┐                   │                     ║
║         │  HIERARCHICAL DECODER  │                   │                     ║
║         │ (Progressive Upsampling)                    │                     ║
║         │ - Decoder_1 (2x up)    │                   │                     ║
║         │ - Decoder_2 (2x up)    │                   │                     ║
║         │ - Boundary Enhancement │                   │                     ║
║         └────────────┬───────────┘                   │                     ║
║                      ↓                                │                     ║
║         ┌─────────────────────────────────────────────┴──────────────┐     ║
║         │                   OUTPUT HEADS (3)                         │     ║
║         │                                                             │     ║
║         ├─ SEG_OUT (Main):     (B, 1, 384, 384) Segmentation         │     ║
║         ├─ EDGE_OUT:           (B, 1, 384, 384) Edge map             │     ║
║         └─ AUX_OUT:            (B, 1, 384, 384) Auxiliary            │     ║
║                                                                       │     ║
║         OUTPUT: Logits (before sigmoid/threshold)                   │     ║
║         LOSS: Multi-task (structure + dice + boundary + ...)        │     ║
║                                                                      │     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. DETAILED CNN STREAM

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            CNN STREAM (Res2Net)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, 3, 384, 384)                                                    │
│    │                                                                         │
│    ├─ Conv(3→48, k=3) + BN + ReLU                                          │
│    └─ Conv(48→48, k=3) + BN + ReLU                                         │
│       ↓                                                                      │
│  CNN_STEM → (B, 48, 384, 384)  [store: 'cnn_stem']                        │
│    │                                                                         │
│    ├─ Res2Net Block (4 branches, multi-scale)                              │
│    │  ├─ 1×1 branch                                                         │
│    │  ├─ 3×3 branch (scale=1)                                              │
│    │  ├─ 3×3 branch (scale=2)                                              │
│    │  └─ 3×3 branch (scale=3)                                              │
│    └─ Concatenate & mix                                                     │
│       ↓                                                                      │
│  RES2NET_1 → (B, 48, 384, 384)  [store: 'res2net1']                       │
│    │                                                                         │
│    ├─ Edge Detection (max_pool - min_pool)                                  │
│    └─ Output: Boundary map                                                  │
│       ↓                                                                      │
│  EDGE_DETECT_1 → (B, 48, 384, 384)  [store: 'edge_detect1']               │
│    │                                                                         │
│    ├─ Res2Net Block (4 branches)                                            │
│    ├─ Stride=2 (2x downsampling)                                            │
│    └─ Output features channels: 48→96                                       │
│       ↓                                                                      │
│  RES2NET_2 → (B, 96, 192, 192)  [store: 'res2net2']                       │
│    │                                                                         │
│    ├─ Edge Detection (max_pool - min_pool)                                  │
│    └─ Output: Boundary map at lower resolution                              │
│       ↓                                                                      │
│  EDGE_DETECT_2 → (B, 96, 192, 192)  [store: 'edge_detect2']               │
│    │                                                                         │
│    ├─ Res2Net Block (4 branches)                                            │
│    ├─ Stride=2 (2x downsampling from previous)                              │
│    └─ Output features channels: 96→192                                      │
│       ↓                                                                      │
│  RES2NET_3 → (B, 192, 96, 96)  [store: 'res2net3']                        │
│                                                                              │
│  KEY FEATURES:                                                              │
│  • Multi-scale receptive fields (4 branches)                                │
│  • No striding on first block (fine detail preservation)                    │
│  • Boundary awareness via edge detection                                    │
│  • Gradual downsampling (×1 → ×2 → ×4)                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. DETAILED TRANSFORMER STREAM

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER STREAM (Swin)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, 3, 384, 384)                                                    │
│    │                                                                         │
│    ├─ Conv(3→96, k=4, s=4) + BN + GELU                                     │
│    │  (Convert image to patches: 384/4 = 96 patches per side)              │
│    │  Channels: 96 (embed_dim)                                              │
│    └─ Tokenization (image→tokens)                                           │
│       ↓                                                                      │
│  PATCH_EMBED → (B, 96, 96, 96)  [store: 'patch_embed']                    │
│    │          [B, 96 channels, 96×96 spatial]                              │
│    │                                                                         │
│    ├─ BasicSwinLayer (2 transformer blocks)                                 │
│    │  ├─ Window attention (window_size=7×7)                                │
│    │  │  └─ Local 7×7 attention (not all-to-all)                           │
│    │  ├─ Shift windows (for cross-window connections)                       │
│    │  ├─ MLP (feed-forward)                                                 │
│    │  └─ LayerNorm + Skip connections                                       │
│    └─ Heads: 3 (embed_dim=96, head_dim=32)                                 │
│       ↓                                                                      │
│  SWIN_1 → (B, 96, 96, 96)  [store: 'swin1']                                │
│    │     [96 channels, 96×96 spatial, shallow transformer]                 │
│    │                                                                         │
│    ├─ Conv(96→192, k=2, s=2) + BN + GELU                                   │
│    │  (Merge neighboring patches: 96 → 48 per side)                        │
│    │  (Double channels: 96 → 192)                                           │
│    └─ Hierarchical downsampling                                             │
│       ↓                                                                      │
│  PATCH_MERGE → (B, 192, 48, 48)  [store: 'patch_merge']                   │
│    │           [192 channels, 48×48 spatial]                               │
│    │                                                                         │
│    ├─ BasicSwinLayer (2 transformer blocks)                                 │
│    │  ├─ Window attention (window_size=7×7)                                │
│    │  ├─ Shift windows                                                      │
│    │  ├─ MLP                                                                 │
│    │  └─ Skip connections                                                   │
│    └─ Heads: 6 (embed_dim=192, head_dim=32)                                │
│       ↓                                                                      │
│  SWIN_3 → (B, 192, 48, 48)  [store: 'swin3']                               │
│           [192 channels, 48×48 spatial, deep transformer]                  │
│                                                                              │
│  KEY FEATURES:                                                              │
│  • Patch embedding (4×4 = 16-pixel patches)                                │
│  • Window-based attention (efficient, local context)                        │
│  • Shift windows (cross-window connections)                                 │
│  • Hierarchical design (coarse→fine)                                        │
│  • Multi-head attention (captures diverse patterns)                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. BOUNDARY-AWARE MIX FUSION (BAMF)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           BOUNDARY-AWARE MIX FUSION (BAMF Layer)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: BAMF_1 (Mid-level Fusion)                                        │
│  ─────────────────────────────────────                                      │
│                                                                              │
│  CNN Input:        x_c2 (B, 96, 192, 192)   ─┐                            │
│  Trans Input:      x_s1 (B, 96, 96, 96)  ─┐  │                            │
│  Edge Guidance:    edge2 (B, 96, 192, 192) │─┤                            │
│                                            │  │                            │
│                      ┌─────────────────────┘  │                            │
│                      ↓                        │                            │
│         ┌─ Conv(96→96): CNN projection      │                            │
│         │ Conv(96→96): Trans projection     │                            │
│         │ Upsample Trans to CNN size        │                            │
│         │                                    │                            │
│         ├─ Concatenate [CNN, Trans]         │                            │
│         │ Conv(192→96) + BN + Sigmoid       │                            │
│         │ → Mix Gate (B, 96, 192, 192)     │                            │
│         │                                    │                            │
│         ├─ Fused = gate⊙CNN + (1-gate)⊙Trans                            │
│         │                                    │                            │
│         ├─ Edge Gate (if edge_feat provided)│                            │
│         │ Conv(96→96): edge projection      │                            │
│         │ → Edge weight (B, 96, 192, 192)   │                            │
│         │                                    │                            │
│         │ Fused = Fused⊙(1+edge_weight)    │                            │
│         │        + CNN⊙edge_weight          │                            │
│         │                                    │                            │
│         ├─ Refinement:                      │                            │
│         │ Conv(96→96, k=3)                  │                            │
│         │ Conv(96→96, k=3)                  │                            │
│         │ SqueezeExcite(96)                 │                            │
│         │                                    │                            │
│         └─ Output = Refined + CNN           │ Residual connection        │
│            (B, 96, 192, 192) ──────────────┘                            │
│                                                                              │
│  FUSED_MID → (B, 96, 192, 192)  [store: 'bamf1_fused_mid']                │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  LEVEL 2: BAMF_2 (Deep Fusion) + Context                                   │
│  ─────────────────────────────────────                                      │
│                                                                              │
│  CNN Input:        x_c3 (B, 192, 96, 96)    ─┐                            │
│  Trans Input:      x_s2 (B, 192, 48, 48) ─┐  │                            │
│  Edge Guidance:    None (deeper layer)      │─┤                            │
│                                             │  │                            │
│         ┌──────────────────────────────────┘  │                            │
│         │ (Same fusion process as BAMF_1)    │                            │
│         │                                     │                            │
│         └─ Output: (B, 192, 96, 96)          │                            │
│            ↓                                  │                            │
│         ┌──────────────────┐                 │                            │
│         │ CONTEXT BLOCK    │ Pyramid context: ┤                            │
│         │ (Multi-scale)    │ Combines features at                          │
│         │ - 1×1 path       │ different scales ┤                            │
│         │ - 3×3 path       │                  │                            │
│         │ - 5×5 path       │                  │                            │
│         └────────┬─────────┘                  │                            │
│                  ↓                             │                            │
│  CONTEXT_BLOCK → (B, 192, 96, 96)             │                            │
│                  [store: 'context_block']    │                            │
│                                              └────────────────────────────│
│                                                                              │
│  WHY BAMF?                                                                  │
│  • CNN: Sharp local details, fast computation                              │
│  • Transformer: Global context, semantic understanding                     │
│  • Edge guidance: Sharpen boundaries, suppress smooth regions              │
│  • Learned blend: Optimal mixing ratio per pixel                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. DECODER (UPSAMPLING PATH)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL DECODER                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CONTEXT_BLOCK Output: (B, 192, 96, 96)                                    │
│    │                                                                         │
│    │  ┌──────────────────────────────────────────┐                        │
│    │  │       DECODER_1 (2x Upsample)          │                        │
│    │  │                                          │                        │
│    │  ├─ Bilinear interpolation: 96→192 spatial│                        │
│    │  │  (B, 192, 96, 96) → (B, 192, 192, 192)│                        │
│    │  │                                          │                        │
│    │  ├─ Skip connection from fused_mid          │                        │
│    │  │  fused_mid: (B, 96, 192, 192)          │                        │
│    │  │                                          │                        │
│    │  ├─ Concatenate: (B, 288, 192, 192)       │                        │
│    │  │                                          │                        │
│    │  ├─ Refinement:                             │                        │
│    │  │  Conv(288→96, k=3)                      │                        │
│    │  │  Conv(96→96, k=3)                       │                        │
│    │  │  SqueezeExcite(96)                      │                        │
│    │  │                                          │                        │
│    │  └─ Output: d1 (B, 96, 192, 192)          │                        │
│    │     [store: 'decoder1']                    │                        │
│    │                                              │                        │
│    │  ┌──────────────────────────────────────────┐                        │
│    │  │       DECODER_2 (2x Upsample)          │                        │
│    │  │                                          │                        │
│    │  ├─ Bilinear interpolation: 192→384       │                        │
│    │  │  (B, 96, 192, 192) → (B, 96, 384, 384)│                        │
│    │  │                                          │                        │
│    │  ├─ Skip connection from x_c1              │                        │
│    │  │  x_c1: (B, 48, 384, 384)               │                        │
│    │  │                                          │                        │
│    │  ├─ Concatenate: (B, 144, 384, 384)      │                        │
│    │  │                                          │                        │
│    │  ├─ Refinement:                             │                        │
│    │  │  Conv(144→48, k=3)                      │                        │
│    │  │  Conv(48→48, k=3)                       │                        │
│    │  │  SqueezeExcite(48)                      │                        │
│    │  │                                          │                        │
│    │  └─ Output: d2 (B, 48, 384, 384)          │                        │
│    │     [store: 'decoder2']                    │                        │
│    │                                              │                        │
│    │  ┌──────────────────────────────────────────┐                        │
│    │  │   BOUNDARY_ENHANCE_BLOCK                │                        │
│    │  │                                          │                        │
│    │  ├─ Input: d2 (B, 48, 384, 384)          │                        │
│    │  ├─ Edge guidance: edge1 (B, 48, 384)    │                        │
│    │  │                                          │                        │
│    │  ├─ Generate boundary map:                 │                        │
│    │  │  max_pool - min_pool (sharpness)        │                        │
│    │  │                                          │                        │
│    │  ├─ Apply edge weighting:                  │                        │
│    │  │  Features ×(1 + edge_weight) at edges   │                        │
│    │  │  Features ×(1 - edge_weight) elsewhere  │                        │
│    │  │                                          │                        │
│    │  ├─ Refinement:                             │                        │
│    │  │  Conv(48→48, k=3)                       │                        │
│    │  │  SqueezeExcite(48)                      │                        │
│    │  │                                          │                        │
│    │  └─ Output: final_feat (B, 48, 384, 384)  │                        │
│    │     [store: 'boundary_enhance']            │                        │
│                                                                              │
│  SKIP CONNECTIONS:                                                          │
│  • Decoder_1 ← fused_mid (mid-level features)                              │
│  • Decoder_2 ← x_c1 (early CNN features)                                   │
│  • Final ← edge1 (boundary guidance)                                        │
│                                                                              │
│  EFFECT:                                        │                        │
│  • Restores spatial resolution progressively                               │
│  • Preserves fine details via skip connections                             │
│  • Enhances boundaries via edge guidance                                    │
│  • Final features: rich, sharp, boundary-aware                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. OUTPUT GENERATION

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT HEADS                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  final_feat: (B, 48, 384, 384)                                              │
│    │                                                                         │
│    ├─ Path 1: SEGMENTATION OUTPUT                                           │
│    │  ├─ Conv(48→1, kernel=1)                                              │
│    │  └─ seg_out: (B, 1, 384, 384) [logits, range: (-∞, +∞)]             │
│    │     [store: 'seg_out']                                                 │
│    │     Post-process: Sigmoid → [0, 1] → Threshold → Binary               │
│    │                                                                         │
│    ├─ Path 2: EDGE OUTPUT (Auxiliary)                                       │
│    │  ├─ Conv(48→1, kernel=1)                                              │
│    │  └─ edge_out: (B, 1, 384, 384) [edge logits]                         │
│    │     [store: 'edge_out']                                                │
│    │     Training: Supervised with computed edge ground truth               │
│    │                                                                         │
│    └─ Path 3: AUXILIARY OUTPUT (from mid-level)                             │
│       ├─ Input: d1 (B, 96, 192, 192)                                       │
│       ├─ Conv(96→1, kernel=1)                                              │
│       ├─ Upsample to final size: (B, 1, 384, 384)                          │
│       └─ aux_out: (B, 1, 384, 384)                                         │
│          [store: 'aux_out']                                                 │
│          Training: Multi-task learning from decoder stage 1                 │
│                                                                              │
│  ═════════════════════════════════════════════════════════════════════════  │
│                                                                              │
│  RETURN: (seg_out, edge_out, aux_out)                                      │
│          All shape: (B, 1, 384, 384)                                       │
│          All in logits (pre-sigmoid)                                        │
│                                                                              │
│  TRAINING LOSS:                                                             │
│  ├─ Main loss: criterion(seg_out, target_mask)                             │
│  ├─ Edge loss: BCE(edge_out, edge_gt)                                      │
│  └─ Auxiliary loss: criterion(aux_out, target_mask)                        │
│     Total = λ_main × loss_main                                             │
│            + λ_edge × loss_edge                                             │
│            + λ_aux × loss_aux                                               │
│                                                                              │
│  INFERENCE POST-PROCESSING:                                                 │
│  ├─ seg_out → Sigmoid → probs [0, 1]                                       │
│  ├─ probs → Threshold (0.45) → binary prediction                           │
│  ├─ Morphological ops: remove small components, holes                       │
│  └─ Keep largest connected component (optional)                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. COMPLETE FORWARD PASS SUMMARY

```
INPUT: (B, 3, 384, 384)
    ↓
PADDING: → (B, 3, 384, 384)  [already multiple of 56]
    ├──────────────────────────────────────────────────────────────┬────────┐
    ↓                                                              ↓        │
CNN STREAM                                        TRANSFORMER       │
├─ cnn_stem                                        ├─ patch_embed    │
│  (B, 48, 384, 384)                              │  (B, 96, 96, 96) │
│                                                  │                 │
├─ res2net1                                       ├─ swin1          │
│  (B, 48, 384, 384)                              │  (B, 96, 96, 96) │
│  └─ edge_detect1                                │                 │
│     (B, 48, 384, 384)                           ├─ patch_merge    │
│                                                  │  (B, 192, 48, 48)│
├─ res2net2                                       │                 │
│  (B, 96, 192, 192)                              ├─ swin3          │
│  └─ edge_detect2                                │  (B, 192, 48, 48)│
│     (B, 96, 192, 192)                           │                 │
│                                                  └──────────┬──────┘
├─ res2net3                                                   │
│  (B, 192, 96, 96)                                          │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
        BOUNDARY-AWARE MIX FUSION (BAMF)
        ├─ bamf1: (B, 96, 192, 192)
        └─ bamf2: (B, 192, 96, 96)
            ↓
        PYRAMID CONTEXT BLOCK
        → (B, 192, 96, 96)
            ↓
        HIERARCHICAL DECODER
        ├─ decoder1: (B, 96, 192, 192)
        ├─ decoder2: (B, 48, 384, 384)
        └─ boundary_enhance: (B, 48, 384, 384)
            ↓
        OUTPUT HEADS (3)
        ├─ seg_out:    (B, 1, 384, 384) ← Main
        ├─ edge_out:   (B, 1, 384, 384) ← Auxiliary
        └─ aux_out:    (B, 1, 384, 384) ← Multi-task
            ↓
        RETURN: (seg_out, edge_out, aux_out)
```

---

## 8. DATA FLOW IN MATRIX FORM

```
Layer                  Input Shape         Output Shape        Params
─────────────────────────────────────────────────────────────────────
cnn_stem              (B, 3, 384, 384)   (B, 48, 384, 384)    3K
res2net1              (B, 48, 384, 384)  (B, 48, 384, 384)    50K
edge_detect1          (B, 48, 384, 384)  (B, 48, 384, 384)    5K
res2net2              (B, 48, 384, 384)  (B, 96, 192, 192)    100K
edge_detect2          (B, 96, 192, 192)  (B, 96, 192, 192)    10K
res2net3              (B, 96, 192, 192)  (B, 192, 96, 96)     150K

patch_embed           (B, 3, 384, 384)   (B, 96, 96, 96)      80K
swin1                 (B, 96, 96, 96)    (B, 96, 96, 96)      200K
patch_merge           (B, 96, 96, 96)    (B, 192, 48, 48)     50K
swin3                 (B, 192, 48, 48)   (B, 192, 48, 48)     450K

bamf1                 Mix of above        (B, 96, 192, 192)    50K
context               (B, 192, 96, 96)   (B, 192, 96, 96)     50K
bamf2                 Mix of above        (B, 192, 96, 96)     50K

decoder1              (B, 192+96,...)     (B, 96, 192, 192)    40K
decoder2              (B, 96+48,...)      (B, 48, 384, 384)    40K
boundary_enhance      (B, 48, 384, 384)  (B, 48, 384, 384)    10K

seg_out               (B, 48, 384, 384)  (B, 1, 384, 384)     1K
edge_out              (B, 48, 384, 384)  (B, 1, 384, 384)     1K
aux_out               (B, 96, 192, 192)  (B, 1, 384, 384)     1K
─────────────────────────────────────────────────────────────────────
TOTAL                                                          ~1.17M
```

---

## 9. KEY DESIGN PRINCIPLES

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRINCIPLE               WHY                          BENEFIT       │
├─────────────────────────────────────────────────────────────────────┤
│ Dual Stream             CNN (fast) + Trans (global)   Best of both  │
│ Hierarchical Fusion     Coarse→Fine progressively    Better context │
│ Edge Guidance           Boundary-aware design         Sharp edges   │
│ Skip Connections        Preserve low-level details   High quality   │
│ Multi-task Learning     Seg + Edge + Aux losses     Regularization │
│ Window Attention        Local self-attention         Efficiency    │
│ Multi-scale Features    Different receptive fields   Robustness    │
│ Gradient Checkpointing  Memory-efficient training    Larger batches │
└─────────────────────────────────────────────────────────────────────┘
```

