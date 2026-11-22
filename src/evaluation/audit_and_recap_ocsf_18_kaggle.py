"""Audit 18 completed Kaggle OCSF runs; write local inspectable paper tables.
Read-only source verification. Does not launch training or use laptop GPU.
"""
import csv, json, hashlib, math, statistics, datetime
from pathlib import Path
from itertools import product

ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
OUT=ROOT/"experiments"/"OCSF_2024_All_18_Train_Runs"
MASTER=OUT/"master16"
EXPECTED_MODELS=["plain","fusion","ordinal","proposed","unet","raunet"]
SEEDS=[42,123,2025]
PILOT_SHA="71e0e6d7e0fa9fa6f541c89f1bfcc57d701bd94730e0508c2dbf45c341de575f"
UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];audit=[];sha_set=set();dates_ref=None
for model,seed in product(EXPECTED_MODELS,SEEDS):
    job=f"{model}_s{seed}"
    folder=OUT/f"ocsf-rainnet-fit-{model}-seed-{seed}" if seed==42 and model in ("plain","fusion") else MASTER
    f={name:folder/f"OCSF_{job}_{name}" for name in ("test_results.json","run_manifest.json","best.weights.h5","last.weights.h5","test_predictions.zip","epoch_history.csv","RUN_COMPLETE.flag","test_confusion_matrix.csv","test_per_scene_metrics.csv")}
    problems=[name+" absent" for name,p in f.items() if not p.is_file() or p.stat().st_size<1]
    try:
        if problems:raise ValueError(", ".join(problems))
        data=json.loads(f["test_results.json"].read_text(encoding="utf8"))
        meta=json.loads(f["run_manifest.json"].read_text(encoding="utf8"))
        assert data["status"]==meta["status"]=="COMPLETE"
        assert data["model"]==meta["variant"]==model and data["seed"]==meta["seed"]==seed
        assert data["test_scene_count"]==20 and data["validation_scene_count"]==19 and data["train_scene_count"]==88
        assert len(data["per_scene"])==20 and len(set(v["date"] for v in data["per_scene"]))==5
        assert len(data["aggregate_test_confusion_matrix"])==5 and all(len(c)==5 for c in data["aggregate_test_confusion_matrix"])
        assert data["rainnet_checkpoint_sha256"]==PILOT_SHA
        assert data["epochs_executed"]>=1 and data["epochs_executed"]<=25
        assert f["best.weights.h5"].stat().st_size>1000
        assert f["test_predictions.zip"].stat().st_size>1000
        assert meta["dataset_archive_sha256"]
        sha_set.add(meta["dataset_archive_sha256"])
        dates=sorted(data["test_UTC_dates"])
        assert len(dates)==5 and dates==sorted(set(v["date"] for v in data["per_scene"]))
        if dates_ref is None:dates_ref=dates
        assert dates==dates_ref,"Test date split differs from other models"
        cm=data["aggregate_test_confusion_matrix"]
        assert sum(sum(z) for z in cm)>1_000_000
        m=data["aggregate_test_metrics"]
        assert all(math.isfinite(m[k]) for k in ("macro_iou","macro_dice","severe_csi","severe_pod","pixel_accuracy"))
        rec={"model":model,"seed":seed,"epochs":data["epochs_executed"],"best_epoch":data["best_epoch"],"parameters":data["parameters"],
           "macro_iou":m["macro_iou"],"macro_dice":m["macro_dice"],"severe_csi":m["severe_csi"],"severe_pod":m["severe_pod"],"severe_far":m["severe_far"],
           "pixel_accuracy":m["pixel_accuracy"],"C0_IoU":m["iou"][0],"C1_IoU":m["iou"][1],"C2_IoU":m["iou"][2],"C3_IoU":m["iou"][3],"C4_IoU":m["iou"][4],
           "test_results":str(f["test_results.json"]),"source":"Kaggle remote, local audited"}
        rows.append(rec)
        audit.append({"run_id":job,"status":"VERIFIED","result_path":str(f["test_results.json"]),"best_weights_bytes":f["best.weights.h5"].stat().st_size,"test_predictions_bytes":f["test_predictions.zip"].stat().st_size})
    except Exception as exc:
        audit.append({"run_id":job,"status":"MISSING_OR_FAILED_QC","reason":repr(exc),"path":str(folder)})
assert len(audit)==18
passed=len(rows)
overall={"generated_utc":UTC,"expected":18,"verified":passed,"missing_or_failed":[x for x in audit if x["status"]!="VERIFIED"],"training_completed_kaggle_reported":18,"dataset_sha256_values":sorted(sha_set),"test_dates":dates_ref,"source_folders":[str(OUT),str(MASTER)],"caveats":["Satellite IMERG V07B is estimated supervisory truth; not independent gauge observations","UTC-date disjoint is not guaranteed meteorological event disjoint","Publication claims require final consistency and pretraining overlap audit"]}
with (OUT/"OCSF_18_Run_Independent_Audit.json").open("w",encoding="utf8") as f:json.dump({"summary":overall,"runs":audit},f,indent=2)
if passed:
    with (OUT/"OCSF_18_Run_Extracted_Metrics.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
if passed==18:
    assert len(sha_set)==1,"Not all runs use same Kaggle dataset hash"
    summary=[]
    for name in EXPECTED_MODELS:
        records=[v for v in rows if v["model"]==name]
        assert len(records)==3 and set(x["seed"] for x in records)==set(SEEDS)
        agg={"model":name,"n":3,"parameters":records[0]["parameters"]}
        for k in ("macro_iou","macro_dice","severe_csi","severe_pod","pixel_accuracy","C3_IoU","C4_IoU"):
            data=[v[k] for v in records]
            agg[k+"_mean"]=statistics.mean(data)
            agg[k+"_sd"]=statistics.stdev(data)
        summary.append(agg)
    with (OUT/"OCSF_18_Run_Model_Mean_SD.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
    fmt=lambda v:f"{v:.4f}"
    lines=["# OCSF-RainNet 18-run independently checked Kaggle output: Results evidence","",f"Generated UTC: {UTC}","",
       "All six models x three seeds (42, 123, 2025) have completed archived experiments; date-held-out split is identical across runs.","",
       "| Model | Parameters | Mean Macro IoU ± SD | Mean Macro Dice ± SD | Mean Severe CSI ± SD | Mean C4 IoU ± SD |","|---|---:|---:|---:|---:|---:|"]
    for v in summary:
        lines.append(f"| {v['model']} | {v['parameters']:,} | {fmt(v['macro_iou_mean'])} ± {fmt(v['macro_iou_sd'])} | {fmt(v['macro_dice_mean'])} ± {fmt(v['macro_dice_sd'])} | {fmt(v['severe_csi_mean'])} ± {fmt(v['severe_csi_sd'])} | {fmt(v['C4_IoU_mean'])} ± {fmt(v['C4_IoU_sd'])} |")
    lines+=["","## Limitations and fair interpretation","",
        "All reported score metrics are against satellite-estimated IMERG V07B, not independent BMKG ground truth.",
        "Dates were partitioned before patch cropping; weather-event independence and pretraining temporal overlap still require audit.",
        "The manuscript Methods and Figures must match calibrated Himawari-9 HSD brightness-temperature input (not historical annotated JPEG).",
        "Small number of held-out dates (five) constrains confidence; do not claim generalized extreme-rain reliability without additional evidence.",
        "Validate predictions and per-class confusions, especially rare severe class C4; macro averages should not obscure failures.",
        "Read the CSV run matrix, ZIP model predictions, best weights and confusion figures before drafting final discussion."]
    (OUT/"OCSF_Results_and_Discussion_Evidence_FINAL.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    overall["report_created"]=True
    (OUT/"OCSF_18_Run_Independent_Audit.json").write_text(json.dumps({"summary":overall,"runs":audit},indent=2),encoding="utf8")
print("INDEPENDENT_AUDIT_VERIFIED",passed,"OF_18",len(overall["missing_or_failed"]),"issues","FINAL_EVIDENCE",passed==18,flush=True)
print("FAILED_IDS",[(x["run_id"],x.get("reason","")) for x in overall["missing_or_failed"]],flush=True)
