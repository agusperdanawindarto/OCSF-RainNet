# Springer SN Computer Science: repository / manuscript status

Target journal: [SN Computer Science](https://link.springer.com/journal/42979/submission-guidelines).

**This research repository is not a journal submission.** The archived Word/PDF manuscript predates the completed calibrated 127-pair experiment. Preserve its Word/Mendeley fields; do not cite the archived manuscript as the final empirical version.

Mandatory editorial adjustments before submission:
1. Replace the outdated claim that only a single rendered-JPEG pilot is available with **127 calibrated AHI HSD/IMERG fields, 32 dates, 88/19/20 train/val/test split**.
2. Revise Figures 2 and 3 and descriptions of the actual AHI B13 physical brightness-temperature input; archived originals contain historic 'not trained' text.
3. Complete structured abstract, keywords, Results, Discussion, Conclusion, reference formatting, actual funding/competing interests/author contributions and data/code availability declarations.
4. Provide full class C0–C4 IoU and Dice, confusion matrices, seed dispersion, per-date severe CSI, runtime, and statistical constraints. Explicitly state that **C4 IoU is almost zero**.
5. Check meteorological-event independence and possible overlap with the pretrained RainNet 2024 source collection (date-disjoint alone is insufficient).
6. Report all results as agreement with IMERG satellite estimates, **not verified rainfall at independent BMKG gauges/radar**.
7. For reused/third-party art ensure permission/credit. Use editable Word and original artwork files in submission.
8. Submit through the journal system with the authors' approval, not by a GitHub repository commit.

The legacy Word document retains native Mendeley citation fields by byte-for-byte copy. A revised submission-ready Word document has **not** been generated here.
