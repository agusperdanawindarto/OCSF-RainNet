# Retrospective event-proximity and historical RainNet pretraining audit

**Audit date:** 22 November 2026

**Manuscript alignment:** OCSF-RainNet V8, Methods §3.1.5, Results §4.9, Table 13.

**Scope:** Retrospective checks of the **unchanged** 127-scene, 18-run experiment; no retraining, resplitting, model selection, or test-score correction.

## Data and checks

The audit compared the frozen [executed date split](../metadata/date_splits_as_executed.csv), raw Himawari-9 HSD segment SHA-256 identifiers, all 127 calibrated input tensors and IMERG target-label tensors, source UTC timestamps, historical RainNet checkpoint identity, archive timestamps and exploratory C3/C4 severe-footprint overlaps. The original satellite input/label binaries are retained in private research storage; this document is a concise public report, not a complete redistributable dataset.

| Retrospective check | Verified result | Evidential boundary |
|---|---|---|
| Source inventory | 127 scenes over 32 UTC dates | Same data as the 18 completed Kaggle runs |
| Date-wise split | Train 88/22, validation 19/5, test 20/5 (scenes/dates) | No exact UTC date crosses partitions |
| Identical raw HSD segment hashes across partitions | **0** | No exact repeated checked raw segments |
| Identical full calibrated input arrays across partitions | **0** | No exact repeated full X tensors |
| Identical full IMERG category target arrays across partitions | **0** | No exact repeated full y tensors |
| Minimum train–test scene time distance | **126 hours (5.25 days)** | Demonstrates time spacing, not independent storms |
| Train–test scene pairs within 72/120 hours | **0 / 0** | Three- and five-day sensitivity windows |
| Train–test scene pairs within 168/240 hours | **26 / 112** | Seven-day exclusion **fails** |
| Train–validation / validation–test minimum separation | **150 / 174 hours** | Temporal proximity only |
| Cross-partition date pairs within 10 days | **17** | Investigated for fixed-grid severe-footprint similarity |
| Maximum coincident C3/C4 union IoU in that date-pair screen | **0.04661** | 27 June train vs 3 July test; **not** storm-track dissimilarity |
| RainNet historical binary-checkpoint SHA-256 | `71e0e6d7e0fa9fa6f541c89f1bfcc57d701bd94730e0508c2dbf45c341de575f` | Matches original archived .h5 and prior [checkpoint provenance](prior_rainnet_pretraining_provenance.md) |
| Historical checkpoint archive modification time | **2023-07-10 23:06:46 UTC** | Predates the May–December 2024 sampled scenes, conditional on archival integrity |
| Historic DATASET 4 training-input HDF5 timestamp | **2023-02-06 UTC** | Consistent chronology; not a complete image-level inventory |

## Interpretation for the manuscript

**Verified:** No exact duplicate checked raw-segment hashes, calibrated input tensors, or target-label tensors across the executed split. Dates do not overlap across train, validation and test. The minimum train–test observation gap is 126 hours. The transferred RainNet checkpoint has the same SHA-256 digest as the archived historic binary model; its recorded filesystem modification date precedes the study observations.

**Evidence-supported but conditional:** The archival checkpoint chronology strongly disfavors direct use of the May–December 2024 study observations in the predecessor checkpoint, **if the recorded historic artifact/timestamps are authentic**. The full original pretraining dataset cannot be reconstructed solely from the checkpoint, and its exact per-item image inventory has not been exhaustively verified.

**Not verified:** Meteorological storm-event independence, persistent/moving system identities across dates, or event-disjoint rerun. A 126-hour gap is not an event-identity certificate; 26 train–test scene pairs fall within a seven-day window. The maximum 0.04661 IoU is coincident fixed-grid C3/C4 mask overlap, not a moving-storm tracking measure. No independent BMKG gauge/radar validation was conducted.

**No outcome changes:** The original six architectures, three random seeds, all 18 GPU runs, held-out scenes, checkpoints and reported metrics were not changed by this retrospective audit. Results quantify agreement with **NASA IMERG V07B satellite precipitation estimates**, not physically verified ground precipitation. C4 detection remains very weak.

## Exact audit materials and stronger validity work

The executed Python audit and structured machine-readable report reside in the authorized local research workspace:

- `D:\DISERTASI_S3\SEGMENTASI_HUJAN\scripts\audit_event_and_pretraining_ocsf.py`
- `D:\DISERTASI_S3\SEGMENTASI_HUJAN\docs\top_tier\EVENT_AND_PRETRAINING_AUDIT_2026-11-22.json`

These are **local provenance references**, not public file links. See also [scene provenance](../metadata/paired_scene_provenance.csv), [18-run audit](../metadata/18_run_independent_audit.json), and [reproducibility protocol](REPRODUCE.md).

To establish event-independent generalization, define meteorological-system identifiers from continuous satellite/IMERG sequences or external tracks, group all scenes by event *before* model selection, and repeat the matched training/test evaluation if events cross existing partitions. Such an audit and rerun have **not** been completed.

**Scientific boundary:** Avoid claims of storm-event-disjoint validation, comprehensive original training-file exclusion, independent BMKG ground truth, statistical significance across events or operational extreme-rain alert reliability.
