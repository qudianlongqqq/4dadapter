# MCVR Stage D2 Prediction Audit

This is a validation-only audit of the fixed D1-B step 2000 checkpoint. No training, checkpoint selection, Gate change, or test access occurred.

| Metric | Value |
|---|---:|
| Residual MAE | 0.008842135647 |
| Residual RMSE | 0.013309672285 |
| Pearson / Spearman | 0.287177287358 / 0.288342003528 |
| Sign accuracy | 0.592938311688 |
| Active-bond precision / recall / F1 | 0.440458218631 / 0.383722729383 / 0.410137677265 |
| Outlier precision / recall / F1 | 0.000000000000 / 0.000000000000 / 0.000000000000 |
| Predicted/target norm ratio | 0.425878714117 |
| Solver achieved/requested ratio | 0.999813820129 |
| Zero-target false-positive rate | 0.294224482383 |
| Confidence ECE | 0.218026676193 |

The edge decoder has weak correlation, incomplete active-bond recall, no outlier recall, and under-confident calibration. Detailed grouped results are in `prediction_quality_summary.csv` and `prediction_calibration.json`.
