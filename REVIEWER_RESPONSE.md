# Response to reviewer comments

This document maps each concern raised in the reviewer's report to the
specific code change in this patch. It is intended as supporting material
for the major revision.

The reviewer's overall judgement was: *"It may be acceptable for M.Tech only
after major revision, provided the student can demonstrate real implementation,
code ownership, ablation studies, and controlled experiments."* What follows
addresses each numbered concern.

---

## 1. The Hausdorff-DT loss does not match the canonical formulation

> *"The implemented Hausdorff-DT loss described in Section 3.7.2 appears
> weaker than the canonical formulation discussed earlier; it uses normalized
> foreground/background distance maps of the ground truth, but does not
> clearly compute the predicted boundary distance transform."*

**Change:** `Utils/losses.py` → `HausdorffDTLoss`.

The previous implementation computed only the GT distance maps. The fixed
class now follows Karimi & Salcudean (2020) Eq. 7,

```
L = (1/|Ω|) Σ_x (p(x) - q(x))^2 * (d_p(x)^α + d_q(x)^α)
```

with `d_p` the distance transform to the *ground-truth* boundary and `d_q`
the distance transform to the *binarised predicted* boundary. Both DTs are
computed under `torch.no_grad()` (DT is not differentiable), and the
gradient flows through the `(p - q)^2` term. A symmetric "distance to
boundary" function (`_signed_boundary_dt`) is provided so that the weight
is well-defined on both sides of the contour.

This is the largest single technical fix; it makes the loss term actually
do what the thesis claims it does.

## 2. No ablation studies

> *"The biggest deficiency is the absence of ablation studies. ... The
> thesis must include at least: Res2Net-only baseline, Swin-only baseline,
> Res2Net + Swin simple concatenation, Res2Net + Swin + BAMF, BAMF without
> edge gating, HD-MixNet without Hausdorff-DT loss, HD-MixNet without
> ED/BE blocks, HD-MixNet with Dice+BCE only, HD-MixNet with boundary loss
> only, Full HD-MixNet."*

**Change:** three pieces, working together.

1. `Models/hd_mixnet.py` now accepts three ablation switches:
   `BRANCH_MODE ∈ {both, cnn_only, swin_only}`, `USE_BAMF`, `USE_EDGE_SUP`.
   When `USE_BAMF` is `False`, the fusion is a simple concat + 1×1 conv
   (this is the "naive concatenation" baseline the reviewer asked for).
   When `USE_EDGE_SUP` is `False`, the ED/BE blocks and the edge head are
   removed entirely. When `BRANCH_MODE` is `cnn_only` or `swin_only`, the
   other branch and its fusion path are skipped.

2. `run_ablations.py` defines exactly the 10 variants the reviewer listed
   (under different but obvious names — see the `ABLATIONS` dict at the top
   of the file) and trains each variant on the same train/val split, with
   the same seed pool, into `checkpoints/ablations/<variant>_seed<N>/`.
   Loss-only ablations (Dice+BCE only, boundary only, no-HD-DT) are
   implemented by zeroing the corresponding `LAMBDA_*` weight.

3. `summarize_ablations.py` walks those checkpoint directories, evaluates
   each on its recorded validation split, and emits a CSV of per-seed
   results and a CSV+Markdown summary with mean ± 95 % CI across seeds.

Running `python run_ablations.py --epochs 50 --seeds 42 1337` followed by
`python summarize_ablations.py` produces a table directly comparable to
the missing "Table 4.x – Ablation results" the reviewer wanted to see.

## 3. The boundary-aware claim is not supported by HD95

> *"The proposed model has worse HD95 than the comparison methods, even
> though HD95 boundary quality is the central motivation of the thesis."*

Two parts to this — the empirical part the reviewer themselves
acknowledged as not strictly controllable, and a methodological
inconsistency in the code that we can fix.

**Methodological inconsistency (now addressed):** threshold selection
during validation maximised Dice every epoch, while the thesis argues the
correct objective is the worst-case boundary error. That is internally
inconsistent. `config.py` now has

```python
THRESHOLD_SELECTION = 'dice'      # 'dice' | 'hd95' | 'composite'
THRESHOLD_COMPOSITE_LAMBDA = 0.05
THRESHOLD_COMPOSITE_HD_NORM = 50.0
```

and `train.py`'s `validate()` honours it. Selecting on `'hd95'` or
`'composite'` (dice minus a scaled HD95 penalty) is what the thesis text
implies; selecting on `'dice'` is what the original code did and is kept
as the default only for backward compatibility. Sweeping this switch is
itself a useful additional ablation.

**Empirical part:** beyond changing the surrogate, the most direct way to
narrow the surrogate-metric gap is to run baselines under the *same*
protocol, which is what the cross-dataset and ablation infrastructure now
support.

## 4. Reproducibility — missing concrete values

> *"Dataset split ratio is not specified clearly... Training configuration
> lacks key hyperparameters: learning rate, batch size, number of epochs
> in text, weight decay, lambda loss weights, ... and random seed. The
> total loss formula gives lambda weights but does not state their
> numerical values."*

**Change:** at the start of every training run, `train.py` now writes
`checkpoints/run_metadata.json` containing:

- the seed,
- the resolved train / val / test sample IDs (so the split is exactly
  reproducible from disk, not just from "VAL_SPLIT = 0.2"),
- every `LAMBDA_*` value, every learning-rate / batch-size / epoch
  hyperparameter, and every model-architecture parameter
  (`CNN_BASE_CHANNELS`, `SWIN_EMBED_DIM`, `SWIN_WINDOW_SIZE`,
  `SWIN_STAGE_DEPTHS`, etc.),
- the threshold-selection mode used,
- the device and (if applicable) the GPU name.

These are exactly the values the thesis prose was missing. The same
metadata is written for every ablation run, so each row of the ablation
table is traceable to a concrete hyperparameter snapshot. `config.py`
itself has been rewritten so the defaults reflect what the thesis
*actually* reports (see also point 8 below) rather than aspirational
"SOTA-beating" settings.

## 5. Res2Net description is incomplete

> *"The Res2Net block description has an incomplete sentence: 'the group
> outputs are' but the formula is missing."*

This is a thesis-text problem, not a code problem; the implementation in
`Models/Components/res2net.py` is complete. The thesis revision should
add the missing equation. For convenience, the formula implemented in the
code is

```
y_1 = x_1
y_i = K_i(x_i + y_{i-1})    for i = 2 .. s
```

where `x_i` is the i-th channel split (out of `s`), `K_i` is the i-th 3×3
convolution, and the final block output is `Concat(y_1, ..., y_s)` followed
by a 1×1 projection and a residual connection. The wording in
Section 3.2.1 should be patched to include this.

## 6. Cross-dataset generalisation

> *"Only Kvasir-SEG is used. No CVC-ClinicDB, CVC-ColonDB, ETIS-Larib,
> or cross-dataset generalization."*

**Change:** `cross_dataset_eval.py` evaluates a single trained checkpoint
on every dataset in `Config.CROSS_DATASETS` whose `images/` and `masks/`
folders exist on disk. Missing datasets are skipped with a logged
warning. The script writes one `results/cross_dataset.json` file
containing mean / std / 95% CI for Dice, IoU and HD95 on every dataset
plus (optionally) one inference-speed measurement.

The dataset paths default to `./Data/<name>/{images,masks}` and can be
overridden via the `KVASIR_DATA_ROOT`, `CVC_CLINIC_ROOT`,
`CVC_COLON_ROOT`, `ETIS_ROOT` environment variables.

## 7. No statistical testing, repeated runs, or confidence intervals

> *"No statistical testing. No repeated runs. No confidence intervals."*

**Change:** `evaluate.py` now collects per-image metrics and reports
mean, std, 95 % CI (normal-approximation), min, median and max. The CI
formula is `1.96 * std / sqrt(n)`. `summarize_ablations.py` does the
same across *seeds* per variant, so the ablation table itself reports
mean ± 95 % CI rather than a single point estimate.

To run the same training multiple times with different seeds:

```bash
python run_ablations.py --only full --seeds 42 1337 2024
```

## 8. No inference time / FPS reported

> *"No inference time or FPS."*

**Change:** `evaluate.py --measure-speed` runs a warmup phase followed by
50 timed forward passes at the configured inference resolution and prints
`ms/image` and `FPS`. `cross_dataset_eval.py --measure-speed` does the
same once per run (not per dataset). On CUDA, `torch.cuda.synchronize()`
is called around the timing window so the measurement is accurate.

## 9. Single qualitative panel, no raw results table or trends

The repository already produced loss / Dice / IoU / HD95 plots
(Section 4.7 of the thesis); the issue is that no raw numbers were
attached. `train.py` now writes `checkpoints/metrics_history.json` after
every validated epoch with `train_loss`, `val_dice`, `val_iou`,
`val_hd95`, `val_threshold` and `learning_rate`. That JSON is the raw
data for the trend figures and should be included as a supplementary file
alongside the figures in the thesis.

## 10. Comparison is not strictly controlled

> *"The thesis acknowledges this weakness and says the comparison is not
> strictly controlled because prior numbers are taken from papers using
> different splits, resolutions, and protocols."*

The cleanest fix here is procedural rather than algorithmic: re-run the
baselines (TransUNet, MixFormer, PraNet) under the same train/val/test
split that `run_metadata.json` now records, at the same input resolution,
with the same evaluation script. This is non-trivial and beyond what this
patch can do automatically, but the protocol is now reproducible: any
baseline trained with this `train.py` (with its model swapped in) will
land on the same split and be evaluated by the same `evaluate.py`.

---

## Files changed or added

| File | Status | Purpose |
|---|---|---|
| `Utils/losses.py` | rewritten | canonical Hausdorff-DT, all losses gated by lambda |
| `Models/hd_mixnet.py` | rewritten | ablation switches (BRANCH_MODE / USE_BAMF / USE_EDGE_SUP) |
| `config.py` | rewritten | honest defaults, ablation switches, concrete lambda values |
| `train.py` | rewritten | run_metadata.json, HD-aware threshold selection, atomic JSON writes |
| `evaluate.py` | rewritten | per-image metrics, 95 % CI, FPS, cross-dataset paths |
| `cross_dataset_eval.py` | new | runs evaluate on all datasets in Config.CROSS_DATASETS |
| `run_ablations.py` | new | trains all variants × seeds the reviewer asked for |
| `summarize_ablations.py` | new | aggregates ablation runs into a results table |
| `docs/REVIEWER_RESPONSE.md` | new | this file |
| `docs/REPRODUCIBILITY.md` | new | concrete numeric hyperparameter table |
| `docs/README_HONEST.md` | new | replacement README with honest reported numbers |

The following files in the repository state overclaimed numbers (e.g.
"Dice 94.5% expected" against the 0.896 actually reported in the thesis).
They should be removed before submission:

- `SOTA_BEATING_GUIDE.md`
- `SOTA_CONFIG_SUMMARY.md`
- `FINAL_SUMMARY_RESULTS.md`
- `EXPECTED_RESULTS.md`
- `START_HERE.md`
- `ALL_DELIVERABLES.txt`

`README_DOCS.md`, `CODE_SUMMARY.md`, `ARCHITECTURE_SUMMARY.md` and
`FLOW_DIAGRAMS.md` contain useful architecture descriptions and can be
kept; the numeric claims inside them (especially "+3-6 % IOU" and
"~1.17M parameters" when the rewritten model uses 48 base channels and
sits at a different size) should be replaced with the numbers that the
new ablation table will produce.
