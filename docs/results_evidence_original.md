# OCSF-RainNet 18-run independently checked Kaggle output: Results evidence

Generated UTC: 2026-11-22T03:40:12.423222+00:00

All six models x three seeds (42, 123, 2025) have completed archived experiments; date-held-out split is identical across runs.

| Model | Parameters | Mean Macro IoU ± SD | Mean Macro Dice ± SD | Mean Severe CSI ± SD | Mean C4 IoU ± SD |
|---|---:|---:|---:|---:|---:|
| plain | 142,637 | 0.2665 ± 0.0213 | 0.3271 ± 0.0393 | 0.0305 ± 0.0529 | 0.0000 ± 0.0000 |
| fusion | 148,234 | 0.2701 ± 0.0081 | 0.3350 ± 0.0143 | 0.0185 ± 0.0290 | 0.0000 ± 0.0000 |
| ordinal | 142,632 | 0.2691 ± 0.0088 | 0.3331 ± 0.0142 | 0.0133 ± 0.0231 | 0.0000 ± 0.0000 |
| proposed | 148,217 | 0.2750 ± 0.0014 | 0.3450 ± 0.0041 | 0.0537 ± 0.0148 | 0.0003 ± 0.0005 |
| unet | 485,853 | 0.2685 ± 0.0011 | 0.3308 ± 0.0022 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| raunet | 508,249 | 0.2648 ± 0.0096 | 0.3228 ± 0.0164 | 0.0027 ± 0.0027 | 0.0000 ± 0.0000 |

## Limitations and fair interpretation

All reported score metrics are against satellite-estimated IMERG V07B, not independent BMKG ground truth.
Dates were partitioned before patch cropping; weather-event independence and pretraining temporal overlap still require audit.
The manuscript Methods and Figures must match calibrated Himawari-9 HSD brightness-temperature input (not historical annotated JPEG).
Small number of held-out dates (five) constrains confidence; do not claim generalized extreme-rain reliability without additional evidence.
Validate predictions and per-class confusions, especially rare severe class C4; macro averages should not obscure failures.
Read the CSV run matrix, ZIP model predictions, best weights and confusion figures before drafting final discussion.
