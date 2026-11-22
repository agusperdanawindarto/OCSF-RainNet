"""Gate and privately stage publication-grade candidate OCSF paired data for Kaggle.
This never runs a model or uses laptop GPU. Scientific and provenance checks are mandatory.
"""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,shutil,subprocess,sys
import numpy as np
ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
LEDGER=ROOT/"metadata/multievent_2024/ahi_pairing_ledger.json"
SPLIT=ROOT/"experiments/2026-11-21_kaggle_agusperdana/OCSF_PLANNED_Date_Disjoint_Splits_NOT_YET_TRAINABLE.csv"
OUT=ROOT/"experiments/2026-11-21_kaggle_agusperdana/OCSF_Kaggle_Data_Readiness.json"
DEST=ROOT/"kaggle/datasets/ocsf-rainnet-calibrated-2024"
stage_upload="--upload" in sys.argv
def publish(q):
    OUT.write_text(json.dumps(q,indent=2),encoding="utf8")
    print("KAGGLE_DATA_QC",json.dumps({k:v for k,v in q.items() if k not in ["errors","source_records"]},default=str)[:5000],flush=True)
data=json.loads(LEDGER.read_text(encoding="utf8")) if LEDGER.exists() else {"scenes":{}}
src={k:v for k,v in data.get("scenes",{}).items() if v.get("status")=="PAIRED_RADIOMETRY_QC" and Path(v["output_path"]).is_file()}
groups={r["utc_date"]:r["assigned_split"] for r in csv.DictReader(SPLIT.open(encoding="utf8"))}
bygroup={k:Counter() for k in ("train","validation","test")}
byday={k:set() for k in bygroup}
issues=[]
for key,rec in src.items():
    path=Path(rec["output_path"])
    try:
        with np.load(path,allow_pickle=False) as z:
            x=z["X"];y=z["y"]
        assert x.shape==(185,460,3) and y.shape==(185,460)
        assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1
        assert rec["valid_BT_fraction"]>=0.99
        g=groups[rec["date_utc"]]
        count=np.bincount(y[y!=255],minlength=5)
        for i,c in enumerate(count):bygroup[g][f"C{i}"]+=int(c)
        byday[g].add(rec["date_utc"])
    except Exception as exc:issues.append({"sample":key,"error":repr(exc)})
counts={g:sum(1 for v in src.values() if groups[v["date_utc"]]==g) for g in bygroup}
gate={
    "project":"OCSF-RainNet","paired_scenes":len(src),"independent_UTC_dates":len(set(v["date_utc"] for v in src.values())),"split_scene_counts":counts,"split_day_counts":{g:len(z) for g,z in byday.items()},"class_counts_per_split":{g:dict(v) for g,v in bygroup.items()},"issues":issues,"training_performed":False,"target":"NASA IMERG Final V07B numerical labels", "input":"Radiometrically calibrated AHI B13 HSD converted to fixed 3-channel brightness-temperature image","source_ethics":"Satellite-imputed targets, NOT independent BMKG truth"
}
checks={
    "all 127 planned paired fields":len(src)==127,
    ">=30 independent UTC days":len(set(v["date_utc"] for v in src.values()))>=30,
    "22/5/5 date-disjoint coverage":{g:len(z) for g,z in byday.items()}=={"train":22,"validation":5,"test":5},
    "all five labels in training":all(bygroup["train"][f"C{i}"]>0 for i in range(5)),
    "rare class represented in validation":bygroup["validation"]["C4"]>0,
    "rare class represented in test":bygroup["test"]["C4"]>0,
    "no quality failures in staged samples":len(issues)==0,
    "all selected paired headers QC":all(rec.get("satpy_start_time") and rec.get("valid_BT_fraction",0)>=0.99 for rec in src.values()),
}
gate["checks"]=checks
gate["eligible_for_kaggle_stage"]=all(checks.values())
if not gate["eligible_for_kaggle_stage"]:
    gate["status"]="BLOCKED_PENDING_PAIR_COMPLETION_OR_QC"
    publish(gate)
    raise SystemExit(0)
if stage_upload:
    previous=json.loads(OUT.read_text(encoding="utf8")) if OUT.exists() else {}
    if previous.get("status") in ("KAGGLE_PRIVATE_DATASET_CREATED","KAGGLE_PRIVATE_DATASET_ALREADY_CREATED"):
        gate["status"]="KAGGLE_PRIVATE_DATASET_ALREADY_CREATED"
        publish(gate)
        raise SystemExit(0)
    DEST.mkdir(parents=True,exist_ok=True)
    for key,rec in src.items():
        name=key+"_B13_BT_IMERG_V07B.npz"
        shutil.copy2(rec["output_path"],DEST/name)
    shutil.copy2(SPLIT,DEST/"split_by_UTC_date.csv")
    shutil.copy2(ROOT/"models/pretrained_rainnet_2024/rainnet_binary_best_05.h5",DEST/"rainnet_binary_best_05.h5")
    shutil.copy2(ROOT/"models/pretrained_rainnet_2024/rainNetModel2labels.py",DEST/"rainNetModel2labels.py")
    shutil.copy2(ROOT/"scripts/rainnet_intensity_models.py",DEST/"rainnet_intensity_models.py")
    shutil.copy2(LEDGER,DEST/"ahi_pairing_ledger.json")
    meta={
        "title":"OCSF RainNet Calibrated AHI IMERG V07 2024",
        "id":"agusperdana/ocsf-rainnet-calibrated-2024",
        "subtitle":"Private paired calibrated Himawari AHI B13 and IMERG V07B fields",
        "description":"Private 2024 Himawari-9 AHI B13 calibrated brightness-temperature grid with matched IMERG Final V07B five-class precipitation estimates. Satellite-estimated labels only; no BMKG independent gauge validation. Study design groups by UTC date before patch extraction; consult all QC manifests before training.",
        "isPrivate":True,"licenses":[{"name":"other"}]
    }
    (DEST/"dataset-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    stage_count=len(list(DEST.glob("*B13_BT_IMERG_V07B.npz")))
    assert stage_count==len(src)
    gate["stage_dir"]=str(DEST);gate["files_staged"]=stage_count;gate["status"]="PRIVATE_DATASET_STAGED_AWAITING_KAGGLE_UPLOAD"
    if "--upload" in sys.argv:
        kaggle=r"C:\Python\Python38\Scripts\kaggle.exe"
        # Upload is create-only. If dataset already exists, record conflict for inspection, no public overwrite.
        result=subprocess.run([kaggle,"datasets","create","-p",str(DEST)],capture_output=True,text=True,timeout=1200)
        gate["kaggle_upload_success"]=result.returncode==0
        gate["kaggle_message"]=(result.stdout+"\n"+result.stderr)[-1500:]
        gate["status"]="KAGGLE_PRIVATE_DATASET_CREATED" if result.returncode==0 else "KAGGLE_UPLOAD_NEEDS_REVIEW"
publish(gate)
