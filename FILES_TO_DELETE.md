# Files to delete from the repository

These files contain claims that are not supported by what the thesis
actually reports (Dice 0.896, HD95 9.8 px on Kvasir-SEG). They should be
removed before submission of the revised thesis or the code is shared
externally.

## Delete

| File | Why |
|---|---|
| `SOTA_BEATING_GUIDE.md` | Claims final Dice 0.945, IoU 0.895, HD95 12.3 px and labels the config "SOTA-beating". Thesis reports 0.896 / 9.8 px. |
| `SOTA_CONFIG_SUMMARY.md` | Same claims; explicitly says "you've beaten SOTA if..." |
| `FINAL_SUMMARY_RESULTS.md` | "PREDICTED FINAL PERFORMANCE: Dice 0.847 ± 0.032 / IOU 0.782 ± 0.041" — these are not the reported numbers. |
| `EXPECTED_RESULTS.md` | Same as above, plus speculative epoch-by-epoch metric trajectories presented as fact. |
| `START_HERE.md` | Repeats the 0.847 / 0.782 prediction. |
| `ALL_DELIVERABLES.txt` | Same. |
| `BACKEND_EXECUTION_GUIDE.md` | Embeds the same predictions. |
| `QUICK_REFERENCE.md` | "Speed: 2-3x", "+3-6% IOU" — never demonstrated. |
| `DELIVERY_SUMMARY.md` | Same. |

`cleanup_code.py` is harmless utility code and can be kept or deleted.

## Keep but edit

| File | Action |
|---|---|
| `README_DOCS.md` | Replace with `docs/README_HONEST.md`. The current version makes the same overclaims. |
| `CODE_SUMMARY.md` | Remove the "Recent Changes & Enhancements" / "Expected Improvements" sections; the architectural description is accurate. |
| `ARCHITECTURE_SUMMARY.md` | Update parameter count and channel widths to match the rewritten `config.py`. |
| `FLOW_DIAGRAMS.md` | Same; the diagrams are accurate up to channel widths. |
| `OPTIMIZATION_CHANGES.md` | Delete; the optimisations it describes (LAMBDA_BOUNDARY 0.40, LAMBDA_HD 0.08) are not what's in the thesis. The actual lambda values are in `docs/REPRODUCIBILITY.md`. |
| `LAYER_TRACKING_GUIDE.md` | Mostly fine; the `Utils/layer_viz.py` it references is *broken* (the class definition is malformed in the uploaded version) and should be repaired separately. |

## Note on `Utils/layer_viz.py`

The uploaded version of `Utils/layer_viz.py` has serious syntax issues:
the `LayerOutputVisualizer` class header is missing, `print_forward_flow`
contains bare strings, and several methods reference undefined variables.
That file does not currently import. None of the patches in this
delivery touch it; the layer-visualisation feature is a separate piece
of work that should be re-done from scratch, or excluded from the
release version of the repository if it is not on the critical path.
