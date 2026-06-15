# Reproducibility — concrete hyperparameter table

The reviewer noted that the thesis text omitted several training and
architecture hyperparameters (concrete λ values, batch size, weight
decay, learning rate, seed, split). These are the values used by
`config.py` in this patch. They are written to
`checkpoints/run_metadata.json` at the start of every training run, so
the on-disk metadata is the ground truth.

| Group | Parameter | Value |
|---|---|---|
| Reproducibility | `SEED` | 42 |
| | Multi-seed eval | 42, 1337, 2024 |
| Data | Dataset | Kvasir-SEG (1000 images) |
| | `VAL_SPLIT` | 0.10 |
| | `TEST_SPLIT` | 0.10 |
| | `IMG_SIZE` | 384 |
| Optimizer | Optimizer | AdamW |
| | `LEARNING_RATE` | 1e-4 |
| | `MIN_LEARNING_RATE` | 1e-6 |
| | `WEIGHT_DECAY` | 1e-4 |
| | LR schedule | Linear warmup → cosine to MIN_LR |
| | `WARMUP_EPOCHS` | 5 |
| | `NUM_EPOCHS` | 100 |
| | `BATCH_SIZE` | 8 (CUDA) / 2 (CPU) |
| | `ACCUMULATION_STEPS` | 1 |
| | `GRAD_CLIP_NORM` | 1.0 |
| Architecture | `CNN_BASE_CHANNELS` | 48 |
| | `RES2NET_SCALE` | 4 |
| | `SWIN_EMBED_DIM` | 96 |
| | `SWIN_WINDOW_SIZE` | 7 |
| | `SWIN_HEADS_STAGE1` | 3 |
| | `SWIN_HEADS_STAGE2` | 6 |
| | `SWIN_STAGE_DEPTHS` | (2, 2) |
| | `SWIN_MLP_RATIO` | 4.0 |
| | `SWIN_DROP_PATH` | 0.1 |
| Loss weights | `LAMBDA_STRUCT` | 1.0 |
| | `LAMBDA_DICE` | 0.5 |
| | `LAMBDA_BCE` | 0.3 |
| | `LAMBDA_BOUNDARY` | 0.30 |
| | `LAMBDA_HD` | 0.10 |
| | `LAMBDA_EDGE` | 0.20 |
| | `LAMBDA_AUX` | 0.40 |
| | `STRUCTURE_POOL_KERNEL` | 31 |
| | `BOUNDARY_LOSS_KERNEL` | 5 |
| | `HD_ALPHA` (α in Karimi & Salcudean Eq. 7) | 2.0 |
| | `HD_BINARIZE_THRESHOLD` | 0.5 |
| | `HD_NORMALIZE` | true |
| Validation | `THRESHOLD_SELECTION` | `dice` (default), `hd95` or `composite` |
| | `THRESHOLD_CANDIDATES` | 0.20, 0.25, …, 0.80 |
| | `THRESHOLD_COMPOSITE_LAMBDA` | 0.05 |
| | `THRESHOLD_COMPOSITE_HD_NORM` | 50.0 px |
| | `VAL_USE_TTA` | false |
| Postprocessing | `USE_POST_PROCESSING` | true |
| | `POST_PROCESS_KERNEL` | 5 |
| | `KEEP_LARGEST_COMPONENT` | true |
| | `MIN_COMPONENT_AREA_RATIO` | 0.001 |
| Inference | `INFERENCE_IMG_SIZE` | 384 |
| | `INFERENCE_BATCH_SIZE` | 1 |
| | `USE_INFERENCE_TTA` | false |

The full **total objective** (`Utils/losses.JointLoss`, Section 3.7.3
of the thesis) with concrete λ values is therefore:

```
L_total = 1.0 * L_struct + 0.5 * L_dice + 0.3 * L_bce
        + 0.30 * L_boundary + 0.10 * L_HD-DT     (main head)
        + 0.40 * L_joint(aux head, same composition)
        + 0.20 * BCE_with_logits(edge head, edge_gt)
```

Setting any λ to 0 disables the corresponding term cleanly, which is how
the loss-ablation variants in `run_ablations.py` are implemented.

## How to reproduce a result

1. Place the Kvasir-SEG `images/` and `masks/` folders under
   `Data/Kvasir/`.
2. `python train.py` — this writes `checkpoints/best_dice_model.pth`,
   `checkpoints/best_hd_model.pth`, `checkpoints/run_metadata.json` and
   `checkpoints/metrics_history.json`.
3. `python evaluate.py --path checkpoints/best_dice_model.pth --measure-speed`
   to report Dice / IoU / HD95 with 95 % CI and the measured FPS.

To reproduce *somebody else's* `run_metadata.json` exactly (same model,
same data split), call the loader with the recorded `train_ids` and
`val_ids` (the `file_names` argument of `KvasirDataset`).

## How to reproduce the ablation table

```bash
python run_ablations.py --epochs 50 --seeds 42 1337 2024
python summarize_ablations.py
```

Outputs land in `results/ablation_table.csv`,
`results/ablation_summary.csv`, and `results/ablation_summary.md`.
