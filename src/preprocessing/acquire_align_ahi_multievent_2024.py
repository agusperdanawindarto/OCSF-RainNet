"""Incremental CPU-only Himawari-9 AHI B13 HSD -> calibrated IMERG-grid pairing.
Input: verified 2024 NASA IMERG V07B source fields (NOAA ERDDAP mirror).
Never uses laptop GPU or performs model training. Event/date holdout gate applied later.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,urllib.request,urllib.parse,time,traceback,sys
import numpy as np,dask
from satpy import Scene
from pyresample import create_area_def

ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
MANIFEST=ROOT/"metadata/multievent_2024/download_manifest.json"
LEDGER=ROOT/"metadata/multievent_2024/ahi_pairing_ledger.json"
HSD=ROOT/"raw/himawari/noaa_ahi_l1b/multievent_2024"
OUT=ROOT/"processed/aligned_multievent_2024"
HSD.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
LIMIT=4
if "--all" in sys.argv: LIMIT=100000
elif "--limit" in sys.argv:LIMIT=int(sys.argv[sys.argv.index("--limit")+1])
source=json.loads(MANIFEST.read_text(encoding="utf8"))
records=[z for day in source["records"] for z in day.get("scenes",[])]
assert len(records)==127
ledger=json.loads(LEDGER.read_text(encoding="utf8")) if LEDGER.exists() else {"schema":1,"model_training":False,"input":"Calibrated NOAA/JMA AHI-L1b B13 HSD","target":"NASA GPM IMERG V07B Final, NOAA ERDDAP","normalization":"fixed grayscale: clip((320 K - BT)/140 K,0,1), 3 replicated channels","resampler":"bilinear","status":"PARTIALLY_PAIRED","scenes":{}}
area=create_area_def("IMERG_Matched_Indonesia",{"proj":"longlat","datum":"WGS84"},width=460,height=185,area_extent=(95.0,-11.5,141.0,7.0),units="degrees")

def flush():
    ledger["updated_utc"]=datetime.now(timezone.utc).isoformat()
    ledger["paired_count"]=sum(1 for r in ledger["scenes"].values() if r.get("status")=="PAIRED_RADIOMETRY_QC")
    ledger["paired_days"]=len({r["date_utc"] for r in ledger["scenes"].values() if r.get("status")=="PAIRED_RADIOMETRY_QC"})
    temp=LEDGER.with_suffix(".json.part")
    temp.write_text(json.dumps(ledger,indent=2),encoding="utf8")
    temp.replace(LEDGER)

def get_hsd(date,hhmm,segment):
    key=f"AHI-L1b-FLDK/{date[:4]}/{date[5:7]}/{date[8:]}/{hhmm}/HS_H09_{date.replace('-','')}_{hhmm}_B13_FLDK_R20_S{segment:02d}10.DAT.bz2"
    d=HSD/date/hhmm;d.mkdir(parents=True,exist_ok=True)
    dest=d/Path(key).name
    url="https://noaa-himawari9.s3.amazonaws.com/"+urllib.parse.quote(key)
    if not dest.exists() or dest.stat().st_size<10000:
        for attempt in range(1,4):
            try:
                tmp=dest.with_suffix(".bz2.part")
                if tmp.exists():tmp.unlink()
                with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"OCSF-RainNet-Research-HSD/1"}),timeout=60) as response,open(tmp,"wb") as f:
                    while True:
                        block=response.read(1<<20)
                        if not block:break
                        f.write(block)
                if tmp.stat().st_size<10000:raise ValueError("Downloaded HSD unexpectedly small")
                tmp.replace(dest);break
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    return dest

processed=0
for sceneinfo in records:
    stamp=sceneinfo["timestamp_start_utc"]
    day=stamp[:10];imerg_hhmm=stamp[11:16]
    sat_time=datetime.strptime(stamp,"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)+timedelta(minutes=10)
    hhmm=sat_time.strftime("%H%M")
    key=day.replace("-","")+"_"+hhmm
    out=OUT/(key+"_B13_BT_IMERG_V07B.npz")
    old=ledger["scenes"].get(key,{})
    if old.get("status")=="PAIRED_RADIOMETRY_QC" and out.exists():continue
    if processed>=LIMIT:break
    log={"date_utc":day,"imerg_start":stamp,"satellite_UTC_expected":sat_time.isoformat(),"segments":[5,6,7],"status":"PROCESSING","output_path":str(out)}
    ledger["scenes"][key]=log
    flush()
    try:
        files=[get_hsd(day,hhmm,seg) for seg in (5,6,7)]
        log["sources"]=[{"name":f.name,"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
        with dask.config.set(scheduler="threads",num_workers=2):
            sat=Scene(reader="ahi_hsd",filenames=[str(f) for f in files])
            sat.load(["B13"])
            actual=sat.start_time
            log["satpy_start_time"]=str(actual)
            if actual.tzinfo is None:actual=actual.replace(tzinfo=timezone.utc)
            if abs((actual-sat_time).total_seconds())>60:
                raise ValueError("Satellite HSD header timestamp differs from expected source time")
            b=sat.resample(area,resampler="bilinear")["B13"]
            bt=np.asarray(b.compute().values,dtype=np.float32)
        assert bt.shape==(185,460)
        assert (str(b.attrs.get("units"))=="K"),"Satpy B13 not in Kelvin"
        data=np.load(sceneinfo["path"],allow_pickle=False)
        y=np.asarray(data["labels"],dtype=np.uint8).copy()
        rate=np.asarray(data["rain_rate_mm_hr"],dtype=np.float32)
        lon=np.asarray(data["lon"],dtype=np.float32);lat=np.asarray(data["lat"],dtype=np.float32)
        assert y.shape==rate.shape==bt.shape
        valid=np.isfinite(bt)&(bt>100)&(bt<400)
        valid_frac=float(np.mean(valid))
        if valid_frac<0.99:raise ValueError("Spatial coverage/BT QC insufficient: "+str(valid_frac))
        yy=y.copy();yy[~valid]=255
        image=np.clip((320.-bt)/140.,0.,1.).astype(np.float32)
        image[~valid]=0.
        X=np.repeat(image[...,None],3,axis=-1)
        assert X.shape==(185,460,3) and np.isfinite(X).all()
        tmp=OUT/(key+"_B13_BT_IMERG_V07B.part.npz")
        np.savez_compressed(tmp,X=X,y=yy,brightness_temperature_K=bt,rain_rate_mm_hr=rate,lon=lon,lat=lat,sat_time_utc=np.array(sat_time.isoformat()),imerg_start_utc=np.array(stamp),source_hsd_sha256=np.array([x["sha256"] for x in log["sources"]]))
        tmp.replace(out)
        log.update({"status":"PAIRED_RADIOMETRY_QC","x_shape":list(X.shape),"y_shape":list(yy.shape),"BT_K_min_max":[float(np.nanmin(bt)),float(np.nanmax(bt))],"valid_BT_fraction":valid_frac,"classes":np.bincount(yy[yy!=255],minlength=5).tolist(),"npz_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"paper_status":"INPUT_PAIR_PRELIMINARY; cross-source spatiotemporal validation and pretraining overlap pending"})
    except Exception as exc:
        log["status"]="QC_OR_DOWNLOAD_FAILED";log["error"]=repr(exc);log["traceback"]=traceback.format_exc()[-1500:]
    ledger["scenes"][key]=log
    flush()
    processed+=1
    print("AHI_PAIR",processed,"/",LIMIT,key,log["status"],log.get("error",""),flush=True)
print("PAIRING_PROGRESS",ledger.get("paired_count"),"scenes",ledger.get("paired_days"),"UTC days",flush=True)
