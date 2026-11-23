# Additional 2026-11-23 architecture comparison: 24 completed runs

The original **18/18** Kaggle experiments (4 RainNet family variants + 2 original external controls, three seeds each) remain **unchanged**. Six separate, audited Kaggle GPU runs added **DeepLabv3+-style** and **SegFormer-B0-style**, each fitted at seeds **42, 123, 2025**. Every model used the same 127 calibrated 2024 Himawari–IMERG pairs and the frozen 88/19/20 scene and 22/5/5 UTC-date split. The test set was **20 scenes**, totaling **1,696,520 valid labeled cells per run**.

The new task-matched models use **random weight initialization**, not the canonical pretrained weights or exact full reference source implementations. The four RainNet-family variants inherit the archived binary RainNet checkpoint. Consequently, a superior mean metric is a **descriptive property of these exact training setups**; it cannot establish inherent architectural or official published-method superiority. Severe-rain C4 IoU is **zero for every seed of both new comparators**.

## Additional run means ± sample SD (three seeds)

| Comparator | Parameters | Macro IoU | Macro Dice | Severe CSI | C3 IoU | C4 IoU |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabv3+-style | 203,349 | 0.2712 ± 0.0039 | 0.3360 ± 0.0074 | 0.0276 ± 0.0162 | 0.02285 ± 0.01274 | 0 |
| SegFormer-B0-style | 3,451,109 | 0.2624 ± 0.0005 | 0.3225 ± 0.0008 | 0.00095 ± 0.00165 | 0.00098 ± 0.00170 | 0 |

OCSF-RainNet retains the largest mean Macro IoU **0.2750 ± 0.0014**, macro Dice **0.3450 ± 0.0041**, and severe CSI **0.0537 ± 0.0148** among all eight implemented configurations. Its C4 IoU remains only **0.00027**.

## Audit and reproducibility

- [Public independent 24-run evidence summary](../results/sota_2026-11-23/OCSF_24_RUN_PUBLIC_AUDIT.json)
- [Eight-model mean ± SD table](../results/sota_2026-11-23/eight_model_mean_sd.csv)
- [Six new run-level test metrics](../results/sota_2026-11-23/new_six_seed_metrics.csv)
- [New source-code architecture definitions](../src/models/ocsf_sota_external_models.py)
- [Six executable Kaggle notebook and metadata folders](../notebooks/kaggle/sota_2026-11-23)
- [Six new confusion matrices](../results/sota_2026-11-23)
- [Original 18-run independent audit](../metadata/18_run_independent_audit.json)
- [V8 retrospective event/provenance audit](EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md)

Run-level metrics were independently recalculated from archived 5×5 test confusion matrices. All six results were verified as COMPLETE, with 20 test scenes, identical five UTC dates, correct run identities, validation-best epochs, all seed/file hashes, and matching 127-pair input SHA-256. No model was selected with test labels and no additional laptop-GPU training was used.

**Boundaries:** The 24 runs share one small-date and satellite-label task. They are not 24 independent storms, are not comparable with published numerical results on different sensors/horizons, do not prove official architecture replication or event-level statistical superiority, and do not establish ground-measured rainfall performance. The main paper's Fig. 2 remains a schematic of the original 18-run six-branch core; Fig. 3 compares the RainNet control with OCSF only. The manuscript V11 updates the combined eight-model numeric comparisons while preserving that provenance.
