# Ablation summary

Numbers are mean over seeds on the validation split recorded
in each run's `run_metadata.json`. The ±values are 95% confidence
intervals across seeds (normal approximation).

| Variant | n seeds | Dice | IoU | HD95 (px) |
|---|---:|---:|---:|---:|
| both_concat | 1 | 0.726 ± 0.000 | 0.624 ± 0.000 | 50.99 ± 0.00 |
| full | 1 | 0.721 ± 0.000 | 0.615 ± 0.000 | 49.41 ± 0.00 |
| full_no_hd_loss | 1 | 0.712 ± 0.000 | 0.611 ± 0.000 | 53.36 ± 0.00 |
| res2net_only | 1 | 0.710 ± 0.000 | 0.601 ± 0.000 | 53.50 ± 0.00 |
