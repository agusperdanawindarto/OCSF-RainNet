"""Recover the three observed slots on 2024-10-11; explicitly preserve missing 20:00 UTC slot."""
from scipy.io import netcdf_file
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,json,hashlib
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
mp=R/"metadata/multievent_2024/download_manifest.json"
manifest=json.loads(mp.read_text(encoding="utf8"))
item=next(x for x in manifest["records"] if x["date_utc"]=="2024-10-11")
nc=R/"raw/imerg_v07/erddap_noaa_2024/2024-10-11.nc"
item["netcdf_sha256"]=hashlib.sha256(nc.read_bytes()).hexdigest()
with netcdf_file(str(nc),"r",mmap=False) as f:
    field=np.array(f.variables["precipitation"][:],dtype="float32")
    time=np.array(f.variables["time"][:],dtype="float64")
    lat=np.array(f.variables["latitude"][:],dtype="float32")[::-1]
    lon=np.array(f.variables["longitude"][:],dtype="float32")
assert field.shape==(3,185,460)
scenes=[]
for idx,(a,epoch) in enumerate(zip(field,time)):
    stamp=datetime.fromtimestamp(epoch,timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert stamp.startswith("2024-10-11T")
    rate=np.flipud(a).copy()
    good=np.isfinite(rate)&(rate>=0)&(rate<1e4)
    y=np.full(rate.shape,255,dtype=np.uint8)
    for label,cond in [(0,rate<1),(1,(rate>=1)&(rate<5)),(2,(rate>=5)&(rate<10)),(3,(rate>=10)&(rate<=20)),(4,rate>20)]:
        y[good&cond]=label
    filename="20241011_"+stamp[11:16].replace(":","")+"_IMERG_2024_V07B.npz"
    target=R/"processed/imerg_multievent_v07"/filename
    np.savez_compressed(target,rain_rate_mm_hr=rate,labels=y,lon=lon,lat=lat,imerg_start_utc=np.array(stamp),source_nc_sha256=np.array(item["netcdf_sha256"]),source_url=np.array(item["url"]))
    c=np.bincount(y[y!=255],minlength=5)
    scenes.append({"timestamp_start_utc":stamp,"path":str(target),"classes":c.tolist(),"valid_cells":int(good.sum()),"invalid_cells":int((~good).sum()),"max_rate_mm_hr":float(np.nanmax(rate[good]))})
item["scenes"]=scenes
item["missing_expected_slots"]=sorted(set(["02:00","08:00","14:00","20:00"])-{x["timestamp_start_utc"][11:16] for x in scenes})
assert item["missing_expected_slots"]==["20:00"],item["missing_expected_slots"]
item["status"]="LABELS_VALIDATED_PARTIAL_3_OF_4"
manifest["summary"]={"days_completed":sum(x["status"].startswith("LABELS_VALIDATED") for x in manifest["records"]),"scenes":sum(len(x.get("scenes",[])) for x in manifest["records"]),"paired_scenes":0,"trainable_pairs":0,"missing_slots":[{"date_utc":"2024-10-11","hours_utc":["20:00"]}]}
mp.write_text(json.dumps(manifest,indent=2),encoding="utf8")
print(json.dumps({"restored":len(scenes),"missing_slots":item["missing_expected_slots"],"summary":manifest["summary"]},indent=2))
