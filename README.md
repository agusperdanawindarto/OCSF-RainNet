# OCSF-RainNet

**Ordinal Cross-Scale Fusion for IMERG-Supervised Rainfall Intensity Segmentation from Himawari Imagery**

Reproducibility package for the **OCSF-RainNet V11.1 audited pre-LaTeX manuscript** intended for *SN Computer Science* (Springer Nature). **127 calibrated Himawari–IMERG pairs, 32 UTC dates, and 24/24 completed and audited Kaggle model×seed experiments (18 original + six supplementary runs across eight configurations).**

> These are comparisons against NASA GPM IMERG Final V07B **satellite-estimated rainfall labels**, not independent BMKG observations. The C4 very-heavy class remains nearly undetected. A retrospective V8 audit ruled out exact checked cross-split digital duplicates and identified a matching pre-2024 historical RainNet checkpoint, but **meteorological-event independence and exhaustive item-level pretraining provenance remain unverified**. See the [event/provenance audit](docs/EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md).

## V11.1 audit-aligned evidence — 23 November 2026 (24 verified Kaggle GPU runs)

The **original 18/18** runs are unchanged: four RainNet factorial variants, U-Net and RA-UNet-style at seeds **42, 123 and 2025**. The supplemental comparison added six independently audited Kaggle GPU runs: **DeepLabv3+-style** and **SegFormer-B0-style**, each at the same three seeds. **All eight configurations use the same calibrated 127-scene Himawari-9 AHI B13/IMERG Final V07B corpus, 88/19/20 scene split, five held-out UTC dates and five-class target definitions.** New results were reconstructed from the per-run 5×5 confusion matrices. The original 18 checkpoints and original plotted experiments were not changed.

| Evaluated model | Parameters | Macro IoU (mean ± SD) | Macro Dice (mean ± SD) | Severe CSI (mean ± SD) | C4 IoU mean |
|---|---:|---:|---:|---:|---:|
| OCSF-RainNet | 148,217 | 0.2750 ± 0.0014 | 0.3450 ± 0.0041 | 0.0537 ± 0.0148 | 0.000270 |
| DeepLabv3+-style | 203,349 | 0.2712 ± 0.0039 | 0.3360 ± 0.0074 | 0.0276 ± 0.0162 | 0.000000 |
| Fusion-only | 148,234 | 0.2701 ± 0.0081 | 0.3350 ± 0.0143 | 0.0185 ± 0.0290 | 0.000000 |
| Ordinal-only | 142,632 | 0.2691 ± 0.0088 | 0.3331 ± 0.0142 | 0.0133 ± 0.0231 | 0.000000 |
| U-Net | 485,853 | 0.2685 ± 0.0011 | 0.3308 ± 0.0022 | 0.0000 ± 0.0000 | 0.000000 |
| RainNet | 142,637 | 0.2665 ± 0.0213 | 0.3271 ± 0.0393 | 0.0305 ± 0.0529 | 0.000000 |
| RA-UNet-style | 508,249 | 0.2648 ± 0.0096 | 0.3228 ± 0.0164 | 0.0027 ± 0.0027 | 0.000000 |
| SegFormer-B0-style | 3,451,109 | 0.2624 ± 0.0005 | 0.3225 ± 0.0008 | 0.0010 ± 0.0016 | 0.000000 |

**Interpretation:** OCSF-RainNet ranks first on *three-seed mean* macro IoU, macro Dice and severe CSI among these eight specifically implemented configurations. DeepLabv3+-style is second on mean macro IoU (0.2712) but below the RainNet control on severe CSI (0.0276 vs 0.0305). SegFormer-B0-style averages 0.2624 macro IoU and 0.0010 severe CSI. Both new architectures have **C4 IoU exactly zero at each seed**; OCSF-RainNet itself remains near zero (mean C4 IoU 0.00027). A RainNet baseline win over OCSF at seed 2025 also precludes a claim of uniform superiority.

**Critical fairness limit:** DeepLabv3+-style (203,349 parameters) and SegFormer-B0-style (3,451,109 parameters) were **independently implemented and randomly initialized**; they are **not official pretrained, code-identical replications** of their original publications. RainNet-derived variants begin from the verified historic transferred checkpoint. Architectural attribution and claims of general SOTA dominance are therefore unjustified. Scores measure agreement with **IMERG satellite-estimated labels**, not BMKG gauge/radar observations, and only five held-out UTC dates are available.

**New inspectable evidence:** [24-run methods and caveats](docs/SOTA_24_RUN_COMPARISON_2026-11-23.md) · [structured validation audit](results/sota_2026-11-23/OCSF_24_RUN_PUBLIC_AUDIT.json) · [8-model mean and SD](results/sota_2026-11-23/eight_model_mean_sd.csv) · [6 per-seed results](results/sota_2026-11-23/new_six_seed_metrics.csv) · [6 test confusion matrices](results/sota_2026-11-23/) · [6 executable Kaggle notebooks](notebooks/kaggle/sota_2026-11-23/) · [external model implementations](src/models/ocsf_sota_external_models.py).

**Figures and journal manuscript:** Fig. 2 **does show the complete eight-model/24-run framework** (18 original and six supplementary Kaggle experiments, one shared held-out partition), while Fig. 3 focuses on the RainNet control and proposed OCSF-RainNet. Manuscript Figs. 4–5 and Tables 4–6 include eight configurations; earlier six-model diagnostics remain explicitly scoped to the 18 original runs. The authoritative V11.1 Word file is maintained separately, not mirrored in this repository. Older DOCX files are historical.

### V11.1 data integrity: native IMERG counts versus HSD-QC paired cells

The audited V11.1 manuscript distinguishes source-grid IMERG category counts from the **valid paired Himawari B13–IMERG labels** actually eligible for training and evaluation. Native IMERG values were not changed or synthesized. The HSD [calibration/pairing implementation](src/preprocessing/acquire_align_ahi_multievent_2024.py) masks targets with label 255 where the matching calibrated B13 brightness temperature fails input QC.

| Data population | Cells | Meaning |
|---|---:|---|
| Source IMERG: 127 × 185 × 460 | **10,807,700** | Original native-rate class inventory before HSD pairing QC |
| HSD input-QC exclusions | **34,798** | 274 invalid input locations per scene (0.322% of the native grid) |
| Valid paired labels, all 127 scenes | **10,772,902** | Eligible paired HSD–IMERG cells after applying the 255 mask |
| Entire held-out source grid | **1,702,000** | 20 scenes × 85,100 native cells |
| **Valid scored test cells per run** | **1,696,520** | 20 scenes × 84,826 valid paired locations |

The [127-scene provenance manifest](metadata/paired_scene_provenance.csv) independently confirms the masked counts. The source-count and paired-valid-count totals refer to **different processing stages of the same scenes**, not different experiments. The published 24 held-out run results, model checkpoints and original figures are unchanged.

**V11.1 Word provenance:** 35-page rendered source, 17 numbered equations, 14 figures, 13 research tables, 17 numbered references and 43 unchanged Mendeley/CSL field instructions. Chen et al. (DeepLabv3+) [16] and Xie et al. (SegFormer) [17] remain **typed citation numbers**, not active Mendeley fields, and need valid BibTeX citation keys in LaTeX. The V11.1 Word source is not currently hosted in the repository. Word page count does not guarantee an identical LaTeX page count.

## Retrospective split integrity and checkpoint provenance (V8 audit, 2026-11-22)

**Why this audit matters:** The manuscript's 18 completed GPU runs used frozen UTC-date-disjoint data partitions. A subsequent, retrospective audit assessed **exact sample leakage**, **temporal proximity**, and **historical checkpoint provenance**; it did **not** change the 127 source–target scenes, train/validation/test split, three seeds, validation-selected weights, or held-out performance scores. It is documented in manuscript **Methods §3.1.5, Results §4.9 and Table 13**.

| Audit question | Evidence from the executed archive | What can be concluded |
|---|---|---|
| Does a UTC date appear in multiple partitions? | **No**; 22/5/5 dates, 88/19/20 scenes in train/validation/test | The frozen split is date-disjoint before crops/augmentation |
| Are identical source segments reused across partitions? | **0** identical checked HSD segment SHA-256 hashes | No exact source-segment duplicates in the audited records |
| Are full input or target arrays repeated across partitions? | **0** identical calibrated input tensor hashes; **0** identical label tensor hashes | No exact duplicated checked full-scene tensors |
| How far apart are train and test observations? | Minimum **126 h (5.25 days)**; **0** train–test scene pairs within 120 h | No direct five-day temporal proximity in the sampled scenes; **not** event separation |
| Does a seven-day embargo hold? | **26** train–test scene pairs within 168 h (**112** within 240 h) | **No**; the stronger seven-day exclusion is not satisfied |
| Are neighboring dates different storm systems? | 17 cross-partition date pairs within 10 days; largest fixed-grid severe-mask C3/C4 union IoU **0.04661** | Diagnostic spatial coincidence only; a moving or persistent storm **cannot** be ruled out |
| Can the transferred RainNet checkpoint be traced to before the study observations? | SHA-256 `71e0e6d7…e575f` matches historical binary model; recorded modification time **2023-07-10 23:06:46 UTC** | Strong archival/chronological evidence against direct reuse of the May–December 2024 study scenes **conditional on archive integrity** |
| Is full pretraining/event independence proven? | Original item-level pretraining inventory and meteorological storm IDs not fully reconstructed | **No**; an event-group-disjoint audit/rerun is still required for event-independent claims |

**Evidence boundary.** These tests rule out **exact digital duplication** among the checked 2024 partitions and strengthen the chronology of transferred pretrained weights. They **do not** establish independent convective systems, exhaustively enumerate every historic pretraining scene, validate rainfall using BMKG gauges/radar, or justify event-independent statistical significance. The rare C4 class remains largely undetected. Results should be cited only as held-out agreement with **NASA IMERG V07B estimated precipitation categories**.

**Audit documentation:** [Full retrospective audit report, results and reproducibility boundaries](docs/EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md) · [Frozen split](metadata/date_splits_as_executed.csv) · [Source-pair provenance](metadata/paired_scene_provenance.csv) · [2024 RainNet checkpoint provenance](docs/prior_rainnet_pretraining_provenance.md).

## Evaluation protocol and interpretation (audited Springer V11.1, 2026-11-23)

**Read this section before citing a score.** The complete evidence is **24 audited trained model–seed evaluations on the same frozen held-out partition**: 18 original factorial/reference comparisons plus six additional external comparisons. The satellite-derived IMERG labels are *reference estimates*, not independent BMKG rain-gauge observations. Main scores are **pixel-pooled across all 20 test scenes within each run**, then **summarized across three seeds**; neither diagnostic crops nor GPU timing measurements substitute for those results.

### What data were evaluated?

| Evaluation item | Executed experiment |
|---|---|
| Input | Calibrated Himawari-9 AHI Band 13 HSD brightness temperature (BT, Kelvin). Input transform: `clip((320-BT)/140,0,1)`; replicated to three identical channels to match pretrained RainNet input |
| Reference labels | Co-timed NASA GPM **IMERG Final V07B** estimated precipitation, discretized into five ordered classes C0–C4 (see Table 3); invalid source labels and HSD-input-QC cells are marked **255** and excluded from loss and metrics |
| Total | **127** radiometrically/time-screened paired scenes over **32 UTC dates** in 2024 |
| Train | **88 scenes / 22 UTC dates** |
| Validation | **19 scenes / 5 UTC dates**; best-checkpoint selection only |
| Test | **20 full scenes / 5 UTC dates**: 2024-05-11, 2024-06-03, 2024-07-03, 2024-08-11, 2024-12-11 |
| Scored domain | Each test scene is **185×460** cells, padded to **192×464** for inference then cropped to original support; **1,696,520 valid paired HSD–IMERG cells per run** (20 × 84,826; 274 excluded by B13 input QC per scene) |
| Split protection | UTC-date grouping **before** training crops/augmentation, identical for all architectures/seeds. Retrospective audit: **0 exact checked source/input/label duplicates** across splits; **126 h minimum train–test gap**. **Event-level independence remains unverified** |

The test partition contains **6,909 C3/C4 cells per run**, including **1,232 C4 cells**. Their rarity makes high overall accuracy insufficient to establish heavy-rain detection. Actual date assignments, paired-scene provenance and model checkpoints are in [the executed split manifest](metadata/date_splits_as_executed.csv), [source-pair audit](metadata/paired_scene_provenance.csv) and [all run artifacts](results/runs). Retrospective source/tensor hashes exclude exact cross-split repetition in checked records; the checkpoint matches a historical binary archive dated July 2023, predating the sampled 2024 observations. These indicators **do not prove storm-event independence or establish a complete item-level original pretraining inventory**.

### Matched comparisons and ablation design

The **2×2 architectural factor design** uses the same five-class target, evaluation split and seeds **42, 123, 2025**:

| Cross-scale fusion | Output head | Repository model | Main question |
|---|---|---|---|
| Off | Multiclass softmax | `plain` | Task-matched multiclass RainNet **control** |
| On | Multiclass softmax | `fusion` | **RQ1:** fusion-only effect |
| Off | Cumulative-exceedance ordinal | `ordinal` | **RQ2:** ordinal-only effect |
| On | Cumulative-exceedance ordinal | `proposed` | **RQ3:** full OCSF-RainNet and component combination |

Additional `unet` and `raunet` runs are task-matched architecture references; **RA-UNet-style** does not assert reproduction of a separate paper's complete experimental protocol. The four RainNet variants start from the designated historical pretrained backbone and the external comparators use their applicable initialization. Shared fitting configuration: 128×128 training crops, batch size 2, maximum 25 epochs, five frozen-backbone warm-up epochs where relevant, early stopping patience 4, **validation-selected best** weights, and a common class-weighted **masked cross-entropy + 0.12 ordinal severity-distance** training objective. The shared loss does **not** isolate the effect of the ordinal head from the loss term: **no loss-only or ungated-fusion ablation was run**. For OCSF-RainNet, best epochs **5, 3, 3** were selected for seeds 42, 123, 2025; this does not establish an improvement from later full-backbone fine-tuning. See [computational protocol](docs/REPRODUCE.md), [per-run manifests](results/runs) and [parameter inventory](tables/Table_04_Model_Parameters.csv).

### Exact quantitative metric definitions

For each trained model–seed run, sum **only valid-label** 5×5 confusion matrices from the **20 held-out scenes** into `M[i,j]`, where **row i is true** C0–C4 and **column j is predicted** C0–C4. For class `k`: `TPk = M[k,k]`; `FNk = sum_j M[k,j] - TPk`; `FPk = sum_i M[i,k] - TPk`.

| Metric | Formula / scoring rule | What it means |
|---|---|---|
| IoU per C0–C4 | `TPk/(TPk+FPk+FNk)` | Exact five-class overlap; **C3 and C4 must be shown separately** |
| Dice per C0–C4 | `2TPk/(2TPk+FPk+FNk)` | Exact class-wise segmentation overlap |
| Macro IoU / Dice | Arithmetic average of the **five class-specific** IoU/Dice scores | C0 prevalence does not weight the macro mean, but can dominate pixel accuracy |
| Pixel accuracy | `sum_k M[k,k]/sum_i,j M[i,j]` | Mainly reflects frequent C0; **not a rare-class detection metric** |
| Severe CSI | `TPs/(TPs+FPs+FNs)` with severe set `S={C3,C4}` | Binary severe-event critical success index (C3 and C4 **combined**) |
| Severe POD | `TPs/(TPs+FNs)` | Severe-event probability of detection / recall |
| Severe FAR | `FPs/(TPs+FPs)` | Fraction of **predicted severe** cells that are false alarms, not the conventional false-positive rate |
| Ordinal MAE — all valid cells | `sum_i,j abs(i-j)*M[i,j]/sum_i,j M[i,j]` | Average absolute **class-index** distance; not physical rainfall-rate MAE |
| Ordinal MAE — true severe only | `sum_(i=3,4),j abs(i-j)*M[i,j]/sum_(i=3,4),j M[i,j]` | Includes **all true C3/C4 cells**, even when predicted as C0–C2 |

For the severe-event binary collapse, `TPs = sum_(i in S,j in S) M[i,j]`, `FNs = sum_(i in S,j not in S) M[i,j]` and `FPs = sum_(i not in S,j in S) M[i,j]`. A C3↔C4 mix-up is a five-class error but *not* a binary severe CSI miss. **Severe CSI therefore cannot be interpreted as C4 IoU or as mean(C3 IoU,C4 IoU).** A zero-denominator metric for a scene is **undefined/N/A**, not a perfect score. Main run-level pooled severe scores have valid denominators. See [metric lineage](metadata/RQ_EVIDENCE_PROVENANCE.json) and [full saved confusion records](results/runs).

### Which averaging level does each table use?

1. **One run:** pool the valid-cell confusion over all 20 held-out scenes, compute the primary scores **once**. [18 original run scores](tables/Table_06_All_18_Runs.csv) and [six additional run results](results/sota_2026-11-23/new_six_seed_metrics.csv) together support the eight-model, 24-run comparisons.
2. **Three seeds:** The [original six-model table](tables/Table_05_Mean_SD_6_Models.csv) covers the 18-run core, while the [eight-model summary](results/sota_2026-11-23/eight_model_mean_sd.csv) gives the **mean ± sample SD (n=3; denominator n−1)** of 24 runs. The ± number is **not** a confidence interval or a test of statistical significance.
3. **Individual scenes:** [Table 10](tables/Table_10_Per_Scene_Results_360_Entries.csv) has **18×20=360** scene-by-run entries. An unweighted scene-level average of CSI generally differs from CSI recomputed after pooling TP/FP/FN counts.
4. **Dates:** [Table 14](tables/Table_14_Heldout_Date_Level_Metrics.csv) reports mean **scene-level** CSI per held-out UTC date across observations/seeds (five dates); it is **not** Table 5's pixel-pooled severe CSI.
5. **Selected images:** revised-manuscript Figures **10–14** use **three post hoc selected 64×96 ROIs, seed 42**. The selected benefit, failure and counterexample illustrate mechanisms and limitations, **not** unbiased model performance. Core factorial comparisons rely on the original 18 runs; the expanded eight-model ranking uses all 24 runs. See [sampling/provenance audit](docs/QUALITATIVE_TEST_PREDICTIONS.md).

### Evaluation conclusions with their actual limitations

- **RQ1 (fusion):** mean Macro IoU **0.2665 → 0.2701**, but severe CSI **0.0305 → 0.0185**; improvement in localized heavy-rain segmentation **was not established**.
- **RQ2 (ordinal):** mean class-distance error on true C3/C4 drops **1.8141 → 1.6098**, but severe CSI **0.0305 → 0.0133**; a lower severity-distance error **does not mean better severe detection**.
- **RQ3 (combined):** OCSF-RainNet reaches mean Macro IoU **0.2750**, Dice **0.3450**, and severe CSI **0.0537**, absolute deltas of **+0.00847, +0.01781, +0.02312** versus RainNet. However, at seed **2025** RainNet is better on Macro IoU and severe CSI, so this is **not a uniform three-seed improvement or a statistical superiority claim**. C4 IoU is only **0.00027** on average and **zero for two seeds**.
- **Resources:** OCSF adds **5,580 parameters (+3.91%)**. The separate **untrained-head GPU graph diagnostic** reports median **64.12 vs 54.86 ms** and TensorFlow allocator peak **43.89 vs 13.96 MiB** for OCSF vs RainNet. These are *not* trained deployment benchmarks, and allocator peak is **not total VRAM**.
- **External validity and audit status:** the retrospective V8 audit found no exact checked cross-split digital duplication and traced the reused checkpoint to an archived July 2023 predecessor (SHA-256 match, chronological interpretation conditional on archive integrity). **Storm-event identities, fully reconstructed historical pretraining-scene lineage, independent BMKG gauge/radar labels, extra geography/year transfer and operational nowcasting/alert validity remain unverified.** Seed SDs and millions of correlated pixels do not replace independent meteorological events.

### Audit trail and manuscript numbering

| Question | Inspectable source |
|---|---|
| All jobs really completed? | [18-run independent audit](metadata/18_run_independent_audit.json), [six additional run metrics](results/sota_2026-11-23/new_six_seed_metrics.csv), [24-run audit](results/sota_2026-11-23/OCSF_24_RUN_PUBLIC_AUDIT.json) |
| Were exact cross-split duplicates, time gaps and checkpoint provenance audited? | [V8 retrospective event/pretraining audit](docs/EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md), [executed dates](metadata/date_splits_as_executed.csv), [checkpoint provenance](docs/prior_rainnet_pretraining_provenance.md) |
| Exactly which evaluation data? | [Executed date split](metadata/date_splits_as_executed.csv), [paired provenance](metadata/paired_scene_provenance.csv) |
| Are C3/C4 being obscured? | [Per-class scores, all seeds](tables/Table_09_Per_Class_Overlap_18_Runs.csv), [18 confusion images](figures/results/confusion) |
| Are RQ claims consistent at each seed? | [Matched-seed deltas](tables/Table_11_RQ_Paired_Seed_Deltas.csv), [RQ contrast summary](tables/Table_13_RQ_Summary_Contrasts.csv) |
| Does ordinal MAE improve on actual missed severe cells? | [18-run ordinal-distance table](tables/Table_08_Ordinal_Distance_Errors_18_Runs.csv), [metric definitions](metadata/RQ_EVIDENCE_PROVENANCE.json) |
| Does severe CSI vary by test date? | [360 scene scores](tables/Table_10_Per_Scene_Results_360_Entries.csv), [five-date summary](tables/Table_14_Heldout_Date_Level_Metrics.csv) |
| Are displayed crops genuine and matched? | [ROI selection audit](docs/QUALITATIVE_TEST_PREDICTIONS.md), [ROI counts](tables/Table_11_Qualitative_ROI_Severe_Error.csv), [matching full-scene scores](tables/Table_12_Matched_Full_Scene_Severe_Metrics.csv) |
| Where is timing documented? | [GPU table](tables/Table_07_GPU_Inference_Profiling.csv), [profiling notebook](notebooks/kaggle/gpu_profiling.ipynb) |

**Avoid duplicate figure/table numbers:** The figures named `figures/results/Figure_10_*`–`Figure_13_*` are older **supplementary repository diagnostics**, **not** the revised-manuscript Figures 10–14. The revised qualitative figures are under [`figures/manuscript_qualitative/`](figures/manuscript_qualitative). Likewise `Table_11_RQ_*` / `Table_12_RQ_*` are supplemental RQ tables, while `Table_11_Qualitative_*` / `Table_12_Matched_*` hold the revised manuscript's qualitative ROI and full-scene evidence. Cite each artifact by descriptive title and exact path, not the number alone.

---

## Figures 1–3: synchronized with the audited Springer V11.1 manuscript

Figure 1 remains the clearly labeled historical January 2024 JPEG pilot. **Figure 2 illustrates all eight configurations and 24 runs in V11.1**; Fig. 3 retains the unchanged RainNet-versus-OCSF architecture. Original six-branch Fig. 2 artwork is preserved under [legacy_18run](figures/manuscript/legacy_18run), and earlier manuscript draft art under [legacy_preV8](figures/manuscript/legacy_preV8). The V11 600-dpi PNG is regenerated from a [vector SVG source](figures/manuscript/Figure_2_V11.svg) and is not asserted to be byte-identical to the Word-embedded PNG. See [V11 figure provenance](metadata/MANUSCRIPT_FIGURE_V11_VERIFICATION.json).

**Figure narratives below follow the revised manuscript:** a short introduction precedes each figure, followed by the caption and an interpretation limited to the completed experiments.

### Figure 1. Dataset characterization of the 7 January 2024 pilot

![Original Figure 1 from the manuscript](figures/manuscript/Figure_1.png)

The original pilot illustrates the archived JPEG, IMERG rate, thresholded five-class target, and class distribution; this is not the input dataset used by the later full calibrated experiment.

### Figure 2. Completed eight-configuration research framework (manuscript V11.1)

Figure 2 traces two independently executed model groups under the same 127 calibrated Himawari-9 HSD–IMERG pairs, frozen date-disjoint partitions and five IMERG label categories: the original RainNet control/factorial ablations plus U-Net and RA-UNet-style (**18 original runs**); and two later task-matched, randomly initialized DeepLabv3+-style and SegFormer-B0-style comparators (**six additional runs**). Every model has three seeds and the same 20 held-out test scenes.

![Figure 2 V11 — shared Himawari–IMERG study framework with eight model branches and 24 completed GPU evaluations](figures/manuscript/Figure_2.png)

*Fig. 2. Research framework of the completed 24-run 2024 calibrated Himawari–IMERG study. The 127 paired scenes, five IMERG rainfall categories, 88/19/20 date-disjoint scene partition and seeds 42/123/2025 are shared across the six original configurations (18 runs) and the two supplementary comparator families (six runs). All eight configurations are evaluated on the same 20 held-out scenes. Event independence, original-pretraining item lineage and ground-measured rainfall validity remain unverified.*

The figure conveys the **actual eight-configuration evidence** and convergent evaluation, **not eight independently collected test datasets or published SOTA checkpoint equivalence**. The first six completed models and 18 runs remain unchanged; the extra six runs are additive. The V8 retrospective [split and provenance audit](docs/EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md) does not establish independent storm-event identities.

[Open Fig. 2 V11 vector source](figures/manuscript/Figure_2_V11.svg) · [original 18-run framework](figures/manuscript/legacy_18run/Figure_2_18run.png) · [V11 figure audit](metadata/MANUSCRIPT_FIGURE_V11_VERIFICATION.json).

### Figure 3. Task-matched RainNet control versus proposed OCSF-RainNet (retained in V11.1)

Figure 3 compares **two separately optimized networks** with the same checkpoint-derived residual RainNet encoder, bridge and decoder. Both receive calibrated Himawari-9 AHI Band 13 brightness-temperature inputs (replicated across three channels for checkpoint compatibility) and the same IMERG V07B five-category targets. The task-matched control uses a five-class softmax head; OCSF-RainNet adds the context-gated cross-scale skip-feature fusion and an ordered cumulative-exceedance output head.

![Figure 3 V8 — architecture of the five-class RainNet control and proposed OCSF-RainNet, with gold proposed modules](figures/manuscript/Figure_3.png)

*Fig. 3. Architecture of the task-matched multiclass RainNet control and OCSF-RainNet, both trained separately using the same calibrated Himawari-9 AHI Band 13 brightness-temperature inputs and NASA IMERG V07B five-category targets. The gold modules denote context-gated cross-scale fusion and the ordered cumulative-exceedance head; pale-blue modules denote inherited RainNet features. Model performance must be read from held-out evaluation, not this schematic.*

Pale-blue boxes indicate inherited features; gold boxes identify the proposed components, **not proven performance gains**. RainNet has **142,637 parameters** versus **148,217** for OCSF-RainNet (+5,580; +3.91%). Fusion-only and ordinal-only ablations, as well as U-Net and RA-UNet-style references, are defined in Fig. 2 and evaluated quantitatively in the held-out results.

[Open Fig. 3 vector source](figures/manuscript/Figure_3_V8.svg) · [Historical pre-V8 Fig. 3](figures/manuscript/legacy_preV8/Figure_3.png).

**Method figure copies and archives:** [calibrated HSD example](figures/methods/Figure_1_Calibrated_2024_Representative_Scene.png), [V11 24-run framework](figures/methods/Figure_2_Research_Framework_Updated.png), and [retained Fig. 3 network schematic](figures/methods/Figure_3_Model_Architecture_Updated.png). The original 18-run Fig. 2 remains in [legacy_18run](figures/manuscript/legacy_18run) and its hash audit has been retained. See [V11 figure verification](metadata/MANUSCRIPT_FIGURE_V11_VERIFICATION.json).

## New held-out image evidence (revised manuscript Figures 10–14, distinct from supplementary result plots)

The original seed-42 qualitative examples remain selected post hoc and are not event-independent results. Figure 1 is the historical pilot, Fig. 2 shows the V11.1 24-run research framework, and Fig. 3 retains the RainNet–OCSF architecture. Legacy figures remain archived.

[Archived 2026-11-22 qualitative DOCX](manuscript/current_revision/OCSF_RainNet_Springer_Qualitative_Revision_2026-11-22.docx) is historical and **not V11.1**. Use the separately provided audited `SN_Computer_Science_OCSF_RainNet_V11_1_PreLaTeX_Audited.docx` for the next LaTeX conversion, including BibTeX keys for references [16] and [17].

![Figure 10 — positive severe detection](figures/manuscript_qualitative/Figure_10_Qualitative_gain.png)

- [Figure 11: all-model failure](figures/manuscript_qualitative/Figure_11_Qualitative_failure.png)
- [Figure 12: fusion-only counterexample](figures/manuscript_qualitative/Figure_12_Qualitative_counterexample.png)
- [Figure 13: true/false positive and missed severe pixels](figures/manuscript_qualitative/Figure_13_Severe_Rain_TP_FP_FN.png)
- [Figure 14: ordinal-distance error maps](figures/manuscript_qualitative/Figure_14_Ordinal_Distance_Error_Maps.png)
- [Table 11: 12 selected-ROI confusion rows](tables/Table_11_Qualitative_ROI_Severe_Error.csv)
- [Table 12: the 12 matched full-scene evaluation rows](tables/Table_12_Matched_Full_Scene_Severe_Metrics.csv)
- [Provenance manifest, case selection, and limitations](docs/QUALITATIVE_TEST_PREDICTIONS.md)

The three examples were selected post hoc, not randomly: a local gain (2024-12-11 20:10 UTC), a complete severe miss (2024-06-03 14:10 UTC), and a fusion-only advantage (2024-08-11 02:10 UTC). All comparisons use the same 64x96-pixel ROI per scene, seed 42, and original IMERG target. Quantitative claims remain based on all held-out scenes and all three seeds. No independent BMKG rain-gauge or radar truth was used.

## RQ1–RQ3: experimentally supported answers

The revised manuscript evaluates **RQ1:** overall and severe-class effects of fusion; **RQ2:** severity-distance versus severe-detection effects of the ordinal head; and **RQ3:** combined-component outcomes across seeds and computational trade-offs. No RQ assumes superiority in advance. All comparisons use the same train/validation/test date split and seeds **42, 123, 2025**.

| RQ | Component vs. RainNet control | Δ mean Macro IoU | Δ mean severe CSI | Δ mean ordinal distance on true C3/C4 cells (lower better) | Improved Macro IoU seeds | Evidence-based verdict |
|---|---|---:|---:|---:|---:|---|
| RQ1 | Fusion-only | +0.0036 | -0.0121 | -0.0791 | 1/3 | Small aggregate gain, **no evidence of better severe-area detection** |
| RQ2 | Ordinal-only | +0.0026 | -0.0172 | -0.2043 | 1/3 | Severe-label distance improves, **severe CSI worsens** |
| RQ3 | OCSF-RainNet | +0.0085 | +0.0231 | -0.1798 | 2/3 | Best mean tested overlap/CSI, **not consistent at all seeds; higher latency** |

The ordinal-distance metrics were **computed from the actual 5×5 test confusion matrices** using mean absolute difference between true/predicted ordinal class indices; the severe-only metric uses true C3/C4 cells, not just correctly detected cells. [Metric definitions and data lineage](metadata/RQ_EVIDENCE_PROVENANCE.json).

### RQ1 — Fusion and localized heavy-rain regions

- **Fusion-only vs RainNet:** Macro IoU **0.2665 → 0.2701**, Macro Dice **0.3271 → 0.3350**. However, severe CSI **0.0305 → 0.0185**, C3 IoU **0.0251 → 0.0145**, and C4 IoU remains **0**.
- Only **1/3 seeds** has higher Macro IoU. Thus a small aggregate overlap improvement **does not establish stronger localized severe-rain segmentation**.
- Inspect [Figure 5](figures/results/Figure_5_Macro_IoU_3_Seeds.png), [Figure 6](figures/results/Figure_6_Severe_CSI_3_Seeds.png), [Figure 9](figures/results/Figure_9_Per_Class_IoU.png) and [paired-seed contrasts](tables/Table_11_RQ_Paired_Seed_Deltas.csv).

### RQ2 — Ordinal head and rainfall-severity errors

- **Ordinal-only vs RainNet:** all-valid-cell mean ordinal class-distance error **0.1080 → 0.1077** and error on true C3/C4 cells **1.8141 → 1.6098**. This is evidence for lower **severity-distance error** under that metric.
- Nevertheless, severe CSI **0.0305 → 0.0133** and C3 IoU **0.0251 → 0.0103**; C4 IoU stays **0**. A better ordinal-distance metric did **not** yield better heavy-rain detection.
- The ordinal head and shared loss were tested; a separate **loss-only ablation** was not performed. See [Figure 10](figures/results/Figure_10_Ordinal_MAE_All_Classes.png), [Figure 11](figures/results/Figure_11_Ordinal_MAE_True_Severe.png) and [Table 8](tables/Table_08_Ordinal_Distance_Errors_18_Runs.csv).

### RQ3 — Combined model: gain, stability and computational trade-off

- **OCSF-RainNet vs RainNet:** mean Macro IoU **0.2665 → 0.2750**, Macro Dice **0.3271 → 0.3450**, severe CSI **0.0305 → 0.0537**, and C3 IoU **0.0251 → 0.0390**.
- The mean gains are **not uniform**: only **2/3 seeds** improve Macro IoU and severe CSI, and C4 IoU remains just **0.00027** on average.
- Parameters rise **142,637 → 148,217** (**+3.91%**). Diagnostic padded-scene GPU median inference time rises **54.86 → 64.12 ms** (**+16.87%**), and TensorFlow allocator peak rises **13.96 → 43.89 MiB**.
- **Verdict:** the combined method leads on the selected **mean** metrics but cannot yet support consistent generalized intense-rain detection or unconditional efficiency. GPU profiling used **untrained heads**, not a deployed trained-model benchmark.
- Inspect [Figure 7](figures/results/Figure_7_Seed_Comparison_Macro_IoU.png), [Figure 8](figures/results/Figure_8_GPU_Inference_Latency.png), [Figure 12](figures/results/Figure_12_RQ_Seedwise_Delta_Macro_IoU.png) and [full RQ contrast tables](tables/Table_13_RQ_Summary_Contrasts.csv).

## Main experimental tables

### Table 3 — Source IMERG class thresholds and HSD-QC paired-data counts

Original IMERG source counts (Table 3 of the manuscript) are shown alongside valid paired counts calculated from the 127-scene provenance manifest.

| Label | Rainfall-rate interval (mm h⁻¹) | Native IMERG cells | Valid HSD–IMERG paired cells | Masked due to B13 QC |
|---|---|---:|---:|---:|
| C0 | 0 ≤ r < 1 | 9,996,003 | 9,964,266 | 31,737 |
| C1 | 1 ≤ r < 5 | 659,836 | 657,323 | 2,513 |
| C2 | 5 ≤ r < 10 | 110,721 | 110,299 | 422 |
| C3 | 10 ≤ r ≤ 20 | 33,808 | 33,712 | 96 |
| C4 | r > 20 | 7,332 | 7,302 | 30 |
| **Total** | | **10,807,700** | **10,772,902** | **34,798** |

Exactly 20 mm h⁻¹ belongs to C3. The source inventory contained zero invalid IMERG labels; after B13 input QC, label **255** excludes 34,798 paired locations. The five rate categories are study-specific, not official BMKG hazard thresholds.


### Table 4 — Architectural parameter inventory

| Model | Parameters | Role |
|---|---:|---|
| Original RainNet binary | 142,617 | Pretraining ancestor |
| plain | 142,637 | Five-class control |
| fusion | 148,234 | Fusion-only ablation |
| ordinal | 142,632 | Ordinal-only ablation |
| proposed | 148,217 | OCSF-RainNet |
| unet | 485,853 | External U-Net |
| raunet | 508,249 | RA-UNet-style |
| deeplabv3plus | 203,349 | DeepLabv3+-style, randomly initialized |
| segformerb0 | 3,451,109 | SegFormer-B0-style, randomly initialized |

### Table 5 — Held-out comparative results (mean ± sample SD, three seeds)

| Model | Macro IoU | Macro Dice | Severe CSI | C3 IoU | C4 IoU |
|---|---:|---:|---:|---:|---:|
| RainNet | 0.2665 ± 0.0213 | 0.3271 ± 0.0393 | 0.0305 ± 0.0529 | 0.0251 ± 0.0434 | 0.0000 ± 0.0000 |
| Fusion-only | 0.2701 ± 0.0081 | 0.3350 ± 0.0143 | 0.0185 ± 0.0290 | 0.0145 ± 0.0233 | 0.0000 ± 0.0000 |
| Ordinal-only | 0.2691 ± 0.0088 | 0.3331 ± 0.0142 | 0.0133 ± 0.0231 | 0.0103 ± 0.0178 | 0.0000 ± 0.0000 |
| OCSF-RainNet | 0.2750 ± 0.0014 | 0.3450 ± 0.0041 | 0.0537 ± 0.0148 | 0.0390 ± 0.0107 | 0.0003 ± 0.0005 |
| U-Net | 0.2685 ± 0.0011 | 0.3308 ± 0.0022 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| RA-UNet-style | 0.2648 ± 0.0096 | 0.3228 ± 0.0164 | 0.0027 ± 0.0027 | 0.0019 ± 0.0020 | 0.0000 ± 0.0000 |
| DeepLabv3+-style | 0.2712 ± 0.0039 | 0.3360 ± 0.0074 | 0.0276 ± 0.0162 | 0.02285 ± 0.01274 | 0.0000 ± 0.0000 |
| SegFormer-B0-style | 0.2624 ± 0.0005 | 0.3225 ± 0.0008 | 0.0010 ± 0.0016 | 0.00098 ± 0.00170 | 0.0000 ± 0.0000 |

### Table 6 — All 24 Kaggle training/evaluation runs (18 original + six supplementary)

| Model | Seed | Best epoch | Macro IoU | Macro Dice | Severe CSI | C3 IoU | C4 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| RainNet | 42 | 24 | 0.2435 | 0.2846 | 0.0000 | 0.0000 | 0.0000 |
| RainNet | 123 | 25 | 0.2702 | 0.3349 | 0.0000 | 0.0000 | 0.0000 |
| RainNet | 2025 | 25 | 0.2857 | 0.3620 | 0.0916 | 0.0752 | 0.0000 |
| Fusion-only | 42 | 19 | 0.2787 | 0.3508 | 0.0519 | 0.0414 | 0.0000 |
| Fusion-only | 123 | 4 | 0.2627 | 0.3230 | 0.0019 | 0.0012 | 0.0000 |
| Fusion-only | 2025 | 3 | 0.2689 | 0.3313 | 0.0016 | 0.0009 | 0.0000 |
| Ordinal-only | 42 | 25 | 0.2760 | 0.3458 | 0.0400 | 0.0308 | 0.0000 |
| Ordinal-only | 123 | 24 | 0.2592 | 0.3178 | 0.0000 | 0.0000 | 0.0000 |
| Ordinal-only | 2025 | 25 | 0.2720 | 0.3357 | 0.0000 | 0.0000 | 0.0000 |
| OCSF-RainNet | 42 | 5 | 0.2765 | 0.3481 | 0.0633 | 0.0463 | 0.0008 |
| OCSF-RainNet | 123 | 3 | 0.2748 | 0.3465 | 0.0611 | 0.0439 | 0.0000 |
| OCSF-RainNet | 2025 | 3 | 0.2736 | 0.3403 | 0.0366 | 0.0268 | 0.0000 |
| U-Net | 42 | 15 | 0.2679 | 0.3306 | 0.0000 | 0.0000 | 0.0000 |
| U-Net | 123 | 9 | 0.2679 | 0.3287 | 0.0000 | 0.0000 | 0.0000 |
| U-Net | 2025 | 17 | 0.2698 | 0.3330 | 0.0000 | 0.0000 | 0.0000 |
| RA-UNet-style | 42 | 11 | 0.2716 | 0.3342 | 0.0027 | 0.0016 | 0.0000 |
| RA-UNet-style | 123 | 11 | 0.2538 | 0.3041 | 0.0000 | 0.0000 | 0.0000 |
| RA-UNet-style | 2025 | 8 | 0.2689 | 0.3302 | 0.0055 | 0.0040 | 0.0000 |
| DeepLabv3+-style | 42 | 8 | 0.2757 | 0.3441 | 0.0269 | 0.0241 | 0.0000 |
| DeepLabv3+-style | 123 | 6 | 0.2692 | 0.3341 | 0.0442 | 0.0349 | 0.0000 |
| DeepLabv3+-style | 2025 | 9 | 0.2686 | 0.3296 | 0.0118 | 0.0095 | 0.0000 |
| SegFormer-B0-style | 42 | 10 | 0.2630 | 0.3233 | 0.0000 | 0.0000 | 0.0000 |
| SegFormer-B0-style | 123 | 11 | 0.2620 | 0.3217 | 0.0029 | 0.0029 | 0.0000 |
| SegFormer-B0-style | 2025 | 6 | 0.2622 | 0.3224 | 0.0000 | 0.0000 | 0.0000 |

### Table 7 — GPU architecture profiling (192×464, batch 1)

| Model | Median time (ms) | P95 (ms) | TensorFlow allocator peak (MiB) | Parameters |
|---|---:|---:|---:|---:|
| RainNet | 54.86 | 58.07 | 13.96 | 142,637 |
| Fusion-only | 65.04 | 70.80 | 44.10 | 148,234 |
| Ordinal-only | 56.95 | 64.37 | 14.90 | 142,632 |
| OCSF-RainNet | 64.12 | 67.00 | 43.89 | 148,217 |
| U-Net | 26.76 | 28.43 | 21.34 | 485,853 |
| RA-UNet-style | 41.13 | 42.34 | 21.43 | 508,249 |

**Scope:** 6 warm-ups + 30 timed passes on a Kaggle GPU, head parameters untrained; allocator peak is not total VRAM and is not an accuracy metric.

### Table 14 — Mean per-scene severe CSI by held-out UTC date

| UTC test date | RainNet | Fusion-only | Ordinal-only | OCSF-RainNet | U-Net | RA-UNet-style |
|---|---:|---:|---:|---:|---:|---:|
| 2024-05-11 | 0.0289 | 0.0090 | 0.0200 | 0.0588 | 0.0000 | 0.0021 |
| 2024-06-03 | 0.0161 | 0.0177 | 0.0034 | 0.0282 | 0.0000 | 0.0007 |
| 2024-07-03 | 0.0116 | 0.0053 | 0.0022 | 0.0223 | 0.0000 | 0.0005 |
| 2024-08-11 | 0.0349 | 0.0254 | 0.0110 | 0.0205 | 0.0000 | 0.0002 |
| 2024-12-11 | 0.0286 | 0.0082 | 0.0141 | 0.0830 | 0.0000 | 0.0088 |

This is the arithmetic mean of scene-level CSI across observations/seeds for a date, **not** the pixel-pooled CSI used in Table 5.

### Complete index of Tables 1–14 and source files

| Table | File | Evidence |
|---|---|---|
| 1 | [Table_01_Prior_Literature.md](tables/Table_01_Prior_Literature.md) | Literature and RainNet lineage |
| 2 | [Table_02_IMERG_127_Source_Fields.csv](tables/Table_02_IMERG_127_Source_Fields.csv) | All 127 IMERG source observations |
| 3a | [Table_03_127_Field_Class_Characterization.json](tables/Table_03_127_Field_Class_Characterization.json) | Source class frequency (127 fields) |
| 3b | [Table_03_Class_Definitions.csv](tables/Table_03_Class_Definitions.csv) | C0–C4 fixed class definitions |
| 4 | [Table_04_Model_Parameters.csv](tables/Table_04_Model_Parameters.csv) | Controlled parameter inventory |
| 5 | [Table_05_Mean_SD_6_Models.csv](tables/Table_05_Mean_SD_6_Models.csv) | Six models, mean and SD |
| 6 | [Table_06_All_18_Runs.csv](tables/Table_06_All_18_Runs.csv) | 18 per-seed result records |
| 7 | [Table_07_GPU_Inference_Profiling.csv](tables/Table_07_GPU_Inference_Profiling.csv) | Two sizes; latency and allocator |
| 8 | [Table_08_Ordinal_Distance_Errors_18_Runs.csv](tables/Table_08_Ordinal_Distance_Errors_18_Runs.csv) | RQ2 per-run severity-distance errors |
| 9 | [Table_09_Per_Class_Overlap_18_Runs.csv](tables/Table_09_Per_Class_Overlap_18_Runs.csv) | Every class IoU/Dice for every run |
| 10 | [Table_10_Per_Scene_Results_360_Entries.csv](tables/Table_10_Per_Scene_Results_360_Entries.csv) | 360 test-scene × run results |
| 11 | [Table_11_RQ_Paired_Seed_Deltas.csv](tables/Table_11_RQ_Paired_Seed_Deltas.csv) | 9 matched-seed RQ contrasts |
| 12 | [Table_12_RQ_All_Metrics_Mean_SD.csv](tables/Table_12_RQ_All_Metrics_Mean_SD.csv) | Metrics, severity error and runtime summary |
| 13 | [Table_13_RQ_Summary_Contrasts.csv](tables/Table_13_RQ_Summary_Contrasts.csv) | Primary RQ deltas and stability counts |
| 14 | [Table_14_Heldout_Date_Level_Metrics.csv](tables/Table_14_Heldout_Date_Level_Metrics.csv) | Test-date variability |

## Supplementary experimental Figures 4–13 (repository plot numbering)

These figures supplement the **unaltered manuscript Figures 1–3** above. Source/target class frequencies are not model predictions; GPU performance charts do not claim trained-model accuracy.

### Figure 4. 127-field IMERG class imbalance

![Figure 4: 127-field IMERG class imbalance](figures/results/Figure_4_IMERG_127_Source_Class_Imbalance.png)

Evidence for Source characterization.

### Figure 5. Macro IoU across six architectures

![Figure 5: Macro IoU across six architectures](figures/results/Figure_5_Macro_IoU_3_Seeds.png)

Evidence for RQ1 / RQ3.

### Figure 6. Severe-rain CSI by architecture

![Figure 6: Severe-rain CSI by architecture](figures/results/Figure_6_Severe_CSI_3_Seeds.png)

Evidence for RQ1 / RQ3.

### Figure 7. Variation across seeds 42 / 123 / 2025

![Figure 7: Variation across seeds 42 / 123 / 2025](figures/results/Figure_7_Seed_Comparison_Macro_IoU.png)

Evidence for RQ3.

### Figure 8. Kaggle GPU graph-inference latency

![Figure 8: Kaggle GPU graph-inference latency](figures/results/Figure_8_GPU_Inference_Latency.png)

Evidence for RQ3: computational cost.

### Figure 9. Held-out class-specific IoU C0–C4

![Figure 9: Held-out class-specific IoU C0–C4](figures/results/Figure_9_Per_Class_IoU.png)

Evidence for All RQs; severe-class limitation.

### Figure 10. All-class ordinal MAE from confusion matrices

![Figure 10: All-class ordinal MAE from confusion matrices](figures/results/Figure_10_Ordinal_MAE_All_Classes.png)

Evidence for RQ2.

### Figure 11. Ordinal MAE conditional on true heavy/very-heavy labels

![Figure 11: Ordinal MAE conditional on true heavy/very-heavy labels](figures/results/Figure_11_Ordinal_MAE_True_Severe.png)

Evidence for RQ2.

### Figure 12. Per-seed Macro IoU deltas versus RainNet

![Figure 12: Per-seed Macro IoU deltas versus RainNet](figures/results/Figure_12_RQ_Seedwise_Delta_Macro_IoU.png)

Evidence for RQ1 / RQ2 / RQ3.

### Figure 13. Severe CSI across five held-out dates

![Figure 13: Severe CSI across five held-out dates](figures/results/Figure_13_Heldout_Date_Severe_CSI.png)

Evidence for All RQs; limited test dates.

## Original 18 confusion matrices and 18 learning curves (six new confusion matrices linked in V11 evidence)

All **36 archived experimental graphics** are shown in expandable galleries. Each is linked to its full-resolution file; corresponding per-run loss histories, confusion CSVs, test metrics and checkpoints remain available.

<details>
<summary><strong>RainNet — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/plain_seed_42.png'><img src='figures/results/confusion/plain_seed_42.png' width='420' alt='RainNet seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/plain_seed_42.png'><img src='figures/results/learning_curves/plain_seed_42.png' width='420' alt='RainNet seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/plain_seed_123.png'><img src='figures/results/confusion/plain_seed_123.png' width='420' alt='RainNet seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/plain_seed_123.png'><img src='figures/results/learning_curves/plain_seed_123.png' width='420' alt='RainNet seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/plain_seed_2025.png'><img src='figures/results/confusion/plain_seed_2025.png' width='420' alt='RainNet seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/plain_seed_2025.png'><img src='figures/results/learning_curves/plain_seed_2025.png' width='420' alt='RainNet seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

<details>
<summary><strong>Fusion-only — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/fusion_seed_42.png'><img src='figures/results/confusion/fusion_seed_42.png' width='420' alt='Fusion-only seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/fusion_seed_42.png'><img src='figures/results/learning_curves/fusion_seed_42.png' width='420' alt='Fusion-only seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/fusion_seed_123.png'><img src='figures/results/confusion/fusion_seed_123.png' width='420' alt='Fusion-only seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/fusion_seed_123.png'><img src='figures/results/learning_curves/fusion_seed_123.png' width='420' alt='Fusion-only seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/fusion_seed_2025.png'><img src='figures/results/confusion/fusion_seed_2025.png' width='420' alt='Fusion-only seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/fusion_seed_2025.png'><img src='figures/results/learning_curves/fusion_seed_2025.png' width='420' alt='Fusion-only seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

<details>
<summary><strong>Ordinal-only — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/ordinal_seed_42.png'><img src='figures/results/confusion/ordinal_seed_42.png' width='420' alt='Ordinal-only seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/ordinal_seed_42.png'><img src='figures/results/learning_curves/ordinal_seed_42.png' width='420' alt='Ordinal-only seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/ordinal_seed_123.png'><img src='figures/results/confusion/ordinal_seed_123.png' width='420' alt='Ordinal-only seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/ordinal_seed_123.png'><img src='figures/results/learning_curves/ordinal_seed_123.png' width='420' alt='Ordinal-only seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/ordinal_seed_2025.png'><img src='figures/results/confusion/ordinal_seed_2025.png' width='420' alt='Ordinal-only seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/ordinal_seed_2025.png'><img src='figures/results/learning_curves/ordinal_seed_2025.png' width='420' alt='Ordinal-only seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

<details>
<summary><strong>OCSF-RainNet — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/proposed_seed_42.png'><img src='figures/results/confusion/proposed_seed_42.png' width='420' alt='OCSF-RainNet seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/proposed_seed_42.png'><img src='figures/results/learning_curves/proposed_seed_42.png' width='420' alt='OCSF-RainNet seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/proposed_seed_123.png'><img src='figures/results/confusion/proposed_seed_123.png' width='420' alt='OCSF-RainNet seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/proposed_seed_123.png'><img src='figures/results/learning_curves/proposed_seed_123.png' width='420' alt='OCSF-RainNet seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/proposed_seed_2025.png'><img src='figures/results/confusion/proposed_seed_2025.png' width='420' alt='OCSF-RainNet seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/proposed_seed_2025.png'><img src='figures/results/learning_curves/proposed_seed_2025.png' width='420' alt='OCSF-RainNet seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

<details>
<summary><strong>U-Net — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/unet_seed_42.png'><img src='figures/results/confusion/unet_seed_42.png' width='420' alt='U-Net seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/unet_seed_42.png'><img src='figures/results/learning_curves/unet_seed_42.png' width='420' alt='U-Net seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/unet_seed_123.png'><img src='figures/results/confusion/unet_seed_123.png' width='420' alt='U-Net seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/unet_seed_123.png'><img src='figures/results/learning_curves/unet_seed_123.png' width='420' alt='U-Net seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/unet_seed_2025.png'><img src='figures/results/confusion/unet_seed_2025.png' width='420' alt='U-Net seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/unet_seed_2025.png'><img src='figures/results/learning_curves/unet_seed_2025.png' width='420' alt='U-Net seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

<details>
<summary><strong>RA-UNet-style — 3 confusion matrices + 3 training curves</strong></summary>

<table><thead><tr><th>Seed</th><th>Test confusion matrix</th><th>Training and validation curves</th></tr></thead><tbody>
<tr><td>42</td><td><a href='figures/results/confusion/raunet_seed_42.png'><img src='figures/results/confusion/raunet_seed_42.png' width='420' alt='RA-UNet-style seed 42 confusion matrix'></a></td><td><a href='figures/results/learning_curves/raunet_seed_42.png'><img src='figures/results/learning_curves/raunet_seed_42.png' width='420' alt='RA-UNet-style seed 42 training curves'></a></td></tr>
<tr><td>123</td><td><a href='figures/results/confusion/raunet_seed_123.png'><img src='figures/results/confusion/raunet_seed_123.png' width='420' alt='RA-UNet-style seed 123 confusion matrix'></a></td><td><a href='figures/results/learning_curves/raunet_seed_123.png'><img src='figures/results/learning_curves/raunet_seed_123.png' width='420' alt='RA-UNet-style seed 123 training curves'></a></td></tr>
<tr><td>2025</td><td><a href='figures/results/confusion/raunet_seed_2025.png'><img src='figures/results/confusion/raunet_seed_2025.png' width='420' alt='RA-UNet-style seed 2025 confusion matrix'></a></td><td><a href='figures/results/learning_curves/raunet_seed_2025.png'><img src='figures/results/learning_curves/raunet_seed_2025.png' width='420' alt='RA-UNet-style seed 2025 training curves'></a></td></tr>
</tbody></table>

</details>

## Code, datasets, availability and journal status

- Actual experiment: 127 paired **calibrated Himawari-9 AHI B13 HSD brightness-temperature scenes** and NASA IMERG Final V07B supervision; 88 training / 19 validation / 20 test scenes, grouped by UTC date.
- Eight configurations, three seeds, 24 audited model–seed runs: [original model source](src), [original Kaggle notebooks](notebooks/kaggle), [original checkpoints](checkpoints), [18 original run-level records](results/runs), [six new notebooks](notebooks/kaggle/sota_2026-11-23), [new model implementations](src/models/ocsf_sota_external_models.py) and [six supplemental run records](results/sota_2026-11-23).
- [NASA IMERG Final V07B DOI](https://doi.org/10.5067/GPM/IMERG/3B-HH/07); [original RainNet study](https://doi.org/10.24507/icicel.18.04.313). The paired 2024 Kaggle dataset is private and raw source data/large prediction ZIPs are not distributed publicly.
- [Audited model results](metadata/18_run_independent_audit.json) · [dataset provenance](metadata/paired_scene_provenance.csv) · [experimental methodology](docs/REPRODUCE.md) · [figure/table index](docs/FIGURES_AND_TABLES_INDEX.md).
- [Archived pre-experiment draft DOCX](manuscript/archived_pre_experiment/OCSF-RainNet_pre_127_field_DRAFT.docx) is retained for historical/Mendeley provenance only and **predates the completed 18-run evidence**. The [2026-11-22 qualitative revision in this repository](manuscript/current_revision/OCSF_RainNet_Springer_Qualitative_Revision_2026-11-22.docx) may predate the subsequent **V8 audit update** (Methods §3.1.5, Results §4.9 and manuscript Table 13). The audited **V11.1 Word manuscript** with all 24 runs and HSD-input-QC denominator reconciliation is supplied separately for LaTeX conversion; old repository DOCX files are not V11.1. Funding/competing-interest declarations, Mendeley fields and submission layout must be checked in the final journal source.
- [Springer readiness checklist](docs/SPRINGER_SUBMISSION_STATUS.md) · [data/code availability](docs/DATA_AND_CODE_AVAILABILITY.md) · [attribution and rights](LICENSE_NOTICE.md).

**Scientific boundary:** Improvements in mean metrics are descriptive for five held-out UTC dates, **not** independent BMKG validation, storm-event-disjoint generalization, or event-level statistical significance. The [V8 audit](docs/EVENT_AND_PRETRAINING_AUDIT_2026-11-22.md) reduces exact-duplication and historical-checkpoint provenance concerns but cannot certify event independence or reconstruct every original pretraining scene. C4 IoU remains close to zero; no loss-only ablation or operational alert certification has been performed.
