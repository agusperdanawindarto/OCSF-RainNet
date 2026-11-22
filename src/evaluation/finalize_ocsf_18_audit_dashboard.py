"""Promote independently verified 18-run science audit to authoritative result summaries.
Preserves historical Kaggle dispatcher log/state as provenance, no GPU training or Kaggle writes.
"""
import json,csv,shutil,datetime,statistics
from pathlib import Path
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
O=R/"experiments/OCSF_2024_All_18_Train_Runs"
a=json.loads((O/"OCSF_18_Run_Independent_Audit.json").read_text(encoding="utf8"))
assert a["summary"]["verified"]==18 and not a["summary"]["missing_or_failed"]
rows=list(csv.DictReader((O/"OCSF_18_Run_Extracted_Metrics.csv").open(encoding="utf8",newline="")))
assert len(rows)==18 and len({(r["model"],r["seed"]) for r in rows})==18
metric=["macro_iou","macro_dice","severe_csi","severe_pod","severe_far","pixel_accuracy"]
numbers=lambda r,k:float(r[k]) if r[k] not in ("", "None", "null") else None
summary=[]
for model in ["plain","fusion","ordinal","proposed","unet","raunet"]:
    subset=[r for r in rows if r["model"]==model]
    assert len(subset)==3 and {r["seed"] for r in subset}=={"42","123","2025"}
    r={"model":model,"completed_seed_count":3,"expected_seed_count":3}
    for k in metric:
        values=[numbers(s,k) for s in subset]
        values=[v for v in values if v is not None]
        r[k+"_mean"]=statistics.mean(values) if values else None
        r[k+"_sd"]=statistics.stdev(values) if len(values)>1 else None
    summary.append(r)
utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
dash={"generated_utc":utc,"status":"COMPLETE_18_OF_18_INDEPENDENTLY_AUDITED","completed":18,"planned":18,"gpu_kaggle_account":"agusperdana",
      "source_note":"16 jobs executed inside Kaggle Master16 notebook; 2 individual Kaggle notebooks plain42/fusion42; all artifacts downloaded to D and independently validated",
      "model_summary":summary,
      "audit_report":str(O/"OCSF_18_Run_Independent_Audit.json"),
      "scientific_caveats":["IMERG satellite target not independent gauge/radar truth","UTC-date disjoint does not guarantee meteorological-event disjoint","Prior pretrained RainNet overlap requires separate audit","Severe-class C4 IoU remains approximately zero; do not overclaim improvement"],
      "dispatcher_state_note":"Legacy training_dispatcher_state.json is a historical snapshot of the earlier sequential dispatcher, superseded by completed Kaggle Master16 run."}
files=["OCSF_training_progress.json","OCSF_all_training_run_metrics.csv","OCSF_all_training_seed_aggregates.csv"]
for name in files:
    dest=O/name
    backup=O/(name+".pre_master16_20261121.bak")
    if dest.exists() and not backup.exists():shutil.copy2(dest,backup)
rows2=[]
for r in rows:
    q={k:r[k] for k in ["model","seed","epochs","best_epoch","parameters","macro_iou","macro_dice","severe_csi","severe_pod","severe_far","pixel_accuracy"]}
    q["run_results"]=r["test_results"];rows2.append(q)
with (O/files[1]).open("w",newline="",encoding="utf8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows2[0]));w.writeheader();w.writerows(rows2)
with (O/files[2]).open("w",newline="",encoding="utf8") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(O/files[0]).write_text(json.dumps(dash,indent=2),encoding="utf8")
print("UPDATED_DASHBOARD",dash["status"],"rows",len(rows),"models",len(summary), "history_dispatcher_preserved",flush=True)
