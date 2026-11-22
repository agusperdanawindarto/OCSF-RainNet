# Genuine held-out qualitative evidence for OCSF-RainNet

Five 600 dpi figures from the original test-prediction NPZ arrays accompany the revised Springer manuscript. Actual AHI Band 13 BT is obtained from calibrated HSD data; target is NASA IMERG Final V07B. The illustration uses seed 42 only. These are post-hoc diagnostic crops, NOT a randomly selected evaluation sample. The existing test split (20 scenes from five dates) and 18-run summaries remain authoritative.

| Case (UTC) | Crop (rows; cols) | True C3/C4 | Severe result in crop |
|---|---|---:|---|
| 2024-12-11 20:10 | 0–63; 75–170 | 287 | Proposed TP133/FP133/FN154, CSI 0.3167; RainNet TP0 |
| 2024-06-03 14:10 | 42–105; 209–304 | 156 | All four RainNet variants TP0, CSI0 |
| 2024-08-11 02:10 | 0–63; 77–172 | 301 | Fusion-only CSI0.2570 > Proposed CSI0.0656 |

No variant correctly classified any of the 60, 25 or 102 target C4 cells in the three selected crops.

The full-scene prediction arrays were compared cell by cell with original stored confusion matrices before cropping. The independently generated ROI confusion data and matching whole-scene metrics are in Tables 11–12. The figure-generation source code is available in src/plotting/generate_ocsf_actual_qualitative_figures.py. The precise source SHA-256 and crop coordinates are in metadata/Qualitative_Test_Scene_Provenance.json.

- [Fig.10 — localized gain](../figures/manuscript_qualitative/Figure_10_Qualitative_gain.png)
- [Fig.11 — all-model failure](../figures/manuscript_qualitative/Figure_11_Qualitative_failure.png)
- [Fig.12 — fusion-only counterexample](../figures/manuscript_qualitative/Figure_12_Qualitative_counterexample.png)
- [Fig.13 — severe-class TP/FP/FN](../figures/manuscript_qualitative/Figure_13_Severe_Rain_TP_FP_FN.png)
- [Fig.14 — ordinal category-distance errors](../figures/manuscript_qualitative/Figure_14_Ordinal_Distance_Error_Maps.png)

The old auxiliary plot filenames Figure_10_* through Figure_13_* in figures/results/ are NOT this revised manuscript figure numbering. The authoritative figures for Sections 4.8 and 5.6 are in figures/manuscript_qualitative/.

The original manuscript's Mendeley field instructions were preserved (44 instructions, 132 Word field nodes and 107 OOXML maths nodes). Refresh ability must still be tested in Microsoft Word with Mendeley Cite. No weather-event-independent or BMKG-ground-validated skill is claimed. The funding, competing-interests and author-contributions declarations remain author-supplied items required before submission.
