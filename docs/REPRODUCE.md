# Computational reproducibility

## Original experiment
- Kaggle account: \`agusperdana\`, private paired dataset: \`agusperdana/ocsf-rainnet-calibrated-2024\`.
- Source: NOAA/JMA calibrated Himawari-9 AHI B13 (brightness temperature in Kelvin) and NASA GPM IMERG Final V07B estimated precipitation.
- Input preprocessing: 0.1° IMERG grid, brightness temperature scaled by \`clip((320K-BT)/140K, 0,1)\`, replicated over three channels for original pretrained RainNet compatibility.
- Five category thresholds: C0 0–<1, C1 1–<5, C2 5–<10, C3 10–20, C4 >20 mm/h. Ignore index 255.
- Date-disjoint samples: 22 UTC dates/88 scenes train; 5/19 validation; 5/20 test. The same dates are used by every model.
- Model variants: RainNet five-class softmax control, fusion-only, ordinal-only, combined OCSF-RainNet, U-Net and task-matched RA-UNet-style.
- Three seeds: 42, 123, 2025.
- Train crop 128×128, batch 2, up to 25 epochs, pretrained RainNet warmup 5 epochs, early-stop patience 4, class-aware masked cross entropy + 0.12 severity-distance.
- Full test scene: 185×460 after padding 192×464 then cropping back; count only valid target cells.
- Run definitions: \`notebooks/kaggle/OCSF_Master_16_Training.ipynb\` (16 runs) and \`ocsf_plain_seed42.ipynb\`, \`ocsf_fusion_seed42.ipynb\` (2 runs).
- The source notebook embeds portable training logic; many standalone scripts in \`src/\` retain original Windows research paths. Adjust those paths before reuse.

## Inspect results without GPU training
Read \`tables/Table_05_Mean_SD_6_Models.csv\`, \`tables/Table_06_All_18_Runs.csv\`, and \`metadata/18_run_independent_audit.json\`. Each \`results/runs/model_seed_n\` folder contains a manifest, per-scene class metrics, confusion matrix, epoch history and full summary. \`checkpoints/best\` contains the best checkpoint for every run.

The provided \`src/evaluation/audit_and_recap_ocsf_18_kaggle.py\` is a local laptop audit for the **original folder structure**. It is not directly path-portable; repository artifact names are normalized for publication. Use archived filenames/records to reconstruct the original run.

## Leakage and remaining validity checks
Splits are UTC-date disjoint, but this alone does not establish that independent meteorological events do not straddle days. Historical RainNet 2024 pretraining source overlap requires a separate audit. The rarest intensity C4 is severely under-predicted. Do not represent these results as operational extreme-rain accuracy or BMKG-validated rainfall rates.
