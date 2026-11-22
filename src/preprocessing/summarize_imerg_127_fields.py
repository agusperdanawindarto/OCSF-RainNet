"""Summarize source-only multi-date IMERG observations for OCSF-RainNet, without model scores."""
from pathlib import Path
from collections import defaultdict
import numpy as np,json,csv
import matplotlib.pyplot as plt
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
M=json.loads((R/"metadata/multievent_2024/download_manifest.json").read_text(encoding="utf8"))
OUT=R/"experiments/2026-11-21_kaggle_agusperdana"
OUT.mkdir(parents=True,exist_ok=True)
records=[];dates=set()
for day in M["records"]:
    for scene in day.get("scenes",[]):
        assert Path(scene["path"]).exists(),scene["path"]
        assert len(scene["classes"])==5
        stamp=scene["timestamp_start_utc"]
        dates.add(day["date_utc"])
        records.append({"date":day["date_utc"],"utc_start":stamp,"class_C0":scene["classes"][0],"class_C1":scene["classes"][1],"class_C2":scene["classes"][2],"class_C3":scene["classes"][3],"class_C4":scene["classes"][4],"valid_cells":scene["valid_cells"],"invalid_cells":scene["invalid_cells"],"max_rate_mm_hr":scene["max_rate_mm_hr"],"label_npz":scene["path"],"himawari_pair":"NOT_YET_VALIDATED"})
assert len(records)==127 and len(dates)==32,(len(records),len(dates))
with open(OUT/"OCSF_IMERG_127_Source_Fields.csv","w",newline="",encoding="utf8") as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
counts=np.array([[r[f"class_C{i}"] for i in range(5)] for r in records],dtype=np.int64)
totals=counts.sum(axis=0)
fraction=totals/totals.sum()
rng=np.random.default_rng(42)
days=sorted(dates);rng.shuffle(days)
groups={"train":days[:22],"validation":days[22:27],"test":days[27:]}
assert [len(groups[x]) for x in groups]==[22,5,5]
with open(OUT/"OCSF_PLANNED_Date_Disjoint_Splits_NOT_YET_TRAINABLE.csv","w",newline="",encoding="utf8") as f:
    w=csv.writer(f);w.writerow(["utc_date","assigned_split","status"])
    for split in groups:
        for day in groups[split]:w.writerow([day,split,"PLANNED_ONLY_AHI_PAIRING_PENDING"])
summary={"research":"OCSF-RainNet","status":"DESCRIPTIVE_LABEL_ONLY_NO_MODEL_RESULTS","time_slots_utc":["02:00","08:00","14:00","20:00"],"date_count":len(dates),"label_fields":len(records),"paired_input_fields":0,"training_executed":False,"total_valid_cells":int(totals.sum()),"sum_class_pixels":totals.tolist(),"class_fraction_percent":(fraction*100).tolist(),"planned_date_split_counts":{k:len(v) for k,v in groups.items()},"missing_source_slots":M["summary"]["missing_slots"],"paired_data_admissibility":"NOT_YET_VERIFIED","main_warning":"IMERG Final V07B satellite estimates are not independent BMKG rainfall observations; NOAA ERDDAP cross-validated against NASA HDF5 for one existing 2024 pilot."}
(OUT/"OCSF_IMERG_127_Characterization.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
fig,ax=plt.subplots(figsize=(9,4.8))
bars=ax.bar([f"C{i}" for i in range(5)],totals)
ax.set_yscale("log")
ax.set_title("IMERG V07B intensity-class distribution across 127 source fields")
ax.set_ylabel("Grid cells, logarithmic scale")
ax.set_xlabel("Rainfall-intensity class (mm/h): C0 <1; C1 1–<5; C2 5–<10; C3 10–20; C4 >20")
for b,c in zip(bars,totals):
    ax.annotate(f"{c:,}\n({100*c/totals.sum():.2f}%)",(b.get_x()+b.get_width()/2,c),ha="center",xytext=(0,5),textcoords="offset points",fontsize=9)
fig.tight_layout()
fig.savefig(OUT/"OCSF_127_IMERG_Source_Characterization_600dpi.png",dpi=600,bbox_inches="tight")
plt.close(fig)
report=f"""# OCSF-RainNet: IMERG source-field characterization (not model results)

- Dataset: NASA GPM IMERG Final V07B via NOAA AOML ERDDAP (one scene cross-validated against original NASA HDF5).
- UTC dates: {len(dates)}; source-only fields: {len(records)}; grid-cell count: {int(totals.sum()):,}.
- Intensity class totals C0–C4: {totals.tolist()}.
- Relative prevalence C0–C4: {[round(v*100,4) for v in fraction]} percent.
- Date-disjoint partition proposal: {len(groups['train'])} train, {len(groups['validation'])} validation, {len(groups['test'])} test dates; **not admissible for training until Himawari georegistration and pair QC**.
- One expected source time is absent at 2024-10-11 20:00 UTC; the other 127 source fields are real, not imputed.
- The evaluation unit is an independent weather event/date, **not a pixel**. These counts alone do not prove independent storm events.
- NOAA source: https://erddap.aoml.noaa.gov/hdb/erddap/griddap/precipitation_2024_v07.html
- NASA canonical data DOI: https://doi.org/10.5067/GPM/IMERG/3B-HH/07

No RainNet, U-Net, RA-UNet-style, or OCSF-RainNet has been trained on these 127 new fields. No test IoU, Dice, CSI, POD, FAR, or generalization claims are supported yet.
"""
(OUT/"OCSF_IMERG_127_Characterization_For_Paper.md").write_text(report,encoding="utf8")
print(json.dumps(summary,indent=2))
