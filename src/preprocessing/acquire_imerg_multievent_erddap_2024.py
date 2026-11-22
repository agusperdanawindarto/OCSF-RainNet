"""Fetch NASA GPM IMERG Final V07B 2024 through public NOAA ERDDAP subset API.
Only creates source/target labels; DOES NOT create fake Himawari inputs and DOES NOT train.
Date-level splits must be defined only after verified Himawari navigation pairing.
"""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np, hashlib,urllib.request,urllib.parse,json,sys,time
from scipy.io import netcdf_file

ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
RAW=ROOT/"raw"/"imerg_v07"/"erddap_noaa_2024"
PROCESSED=ROOT/"processed"/"imerg_multievent_v07"
METADATA=ROOT/"metadata"/"multievent_2024"
for p in (RAW,PROCESSED,METADATA):p.mkdir(parents=True,exist_ok=True)
SOURCE="https://erddap.aoml.noaa.gov/hdb/erddap/griddap/precipitation_2024_v07.nc"
DATES=[f"2024-{month:02d}-{day:02d}" for month in range(5,13) for day in (3,11,19,27)]
TIMES=["02:00","08:00","14:00","20:00"]
SINGLE_DAY = "--pilot" in sys.argv
if SINGLE_DAY:DATES=DATES[:1]
manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"research":"OCSF-RainNet","source":SOURCE,"target_product":"NASA GPM_3IMERGHH Final V07B","status":"PRECIPITATION_LABELS_ONLY_NO_HIMAWARI_PAIRING","dates_requested":DATES,"hours_utc":TIMES,"rain_classes":"C0 0<=r<1; C1 1<=r<5; C2 5<=r<10; C3 10<=r<=20; C4 r>20 mm/h","source_validation":"Pilot 2024-01-07 matched NASA HDF5 to <=3.06e-5 mm/h across 85100 cells, flip latitude required.","records":[]}

def save_manifest():
    (METADATA/"download_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")

for dstr in DATES:
    item={"date_utc":dstr,"source_type":"IMERG Final V07B from NOAA ERDDAP","paired_himawari":False}
    dest=RAW/(dstr+".nc")
    expr=(f"precipitation[({dstr}T02:00:00Z):12:({dstr}T20:00:00Z)]"
          "[(-11.45):1:(6.95)][(95.05):1:(140.95)]")
    url=SOURCE+"?"+urllib.parse.quote(expr,safe="[]():,.-")
    item["url"]=url
    if not dest.exists():
        for attempt in range(1,4):
            try:
                temp=dest.with_suffix(".nc.part")
                with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"OCSF-RainNet-scientific-research"}),timeout=90) as resp, open(temp,"wb") as f:
                    while True:
                        b=resp.read(1<<20)
                        if not b:break
                        f.write(b)
                if temp.stat().st_size<500_000:raise ValueError("Unexpected small NetCDF response")
                temp.replace(dest)
                break
            except Exception as exc:
                item["last_error"]=repr(exc)
                if attempt==3:item["status"]="FAILED_DOWNLOAD"
                time.sleep(1)
    if dest.exists():
        try:
            with netcdf_file(str(dest),mode="r",mmap=False) as f:
                raw=np.asarray(f.variables["precipitation"][:],dtype="float32")
                lat=np.asarray(f.variables["latitude"][:],dtype="float32")
                lon=np.asarray(f.variables["longitude"][:],dtype="float32")
                t=np.asarray(f.variables["time"][:])
            if raw.shape!=(4,185,460):raise ValueError("IMERG grid shape/time mismatch "+str(raw.shape))
            if not np.all(np.diff(lat)>0) or not np.all(np.diff(lon)>0):raise ValueError("Native grids not ascending")
            if not (abs(float(lat[0])+11.45)<0.001 and abs(float(lat[-1])-6.95)<0.001 and abs(float(lon[0])-95.05)<0.001 and abs(float(lon[-1])-140.95)<0.001):
                raise ValueError("unexpected NOAA ERDDAP spatial extent")
            item["netcdf_sha256"]=hashlib.sha256(dest.read_bytes()).hexdigest()
            for ts,a in zip(TIMES,raw):
                rate=np.flipud(a).copy()
                valid=np.isfinite(rate)&(rate>=0)&(rate<1e4)
                y=np.full(rate.shape,255,dtype=np.uint8)
                y[valid&(rate<1)]=0
                y[valid&(rate>=1)&(rate<5)]=1
                y[valid&(rate>=5)&(rate<10)]=2
                y[valid&(rate>=10)&(rate<=20)]=3
                y[valid&(rate>20)]=4
                c=np.bincount(y[y!=255],minlength=5)
                name=dstr.replace("-","")+"_"+ts.replace(":","")
                out=PROCESSED/(name+"_IMERG_2024_V07B.npz")
                np.savez_compressed(out,rain_rate_mm_hr=rate,labels=y,lon=lon,lat=lat[::-1].copy(),imerg_start_utc=np.array(dstr+"T"+ts+":00Z"),source_nc_sha256=np.array(item["netcdf_sha256"]),source_url=np.array(url))
                item.setdefault("scenes",[]).append({"timestamp_start_utc":dstr+"T"+ts+":00Z","path":str(out),"classes":c.tolist(),"valid_cells":int(valid.sum()),"max_rate_mm_hr":float(np.nanmax(rate[valid])) if valid.any() else None,"invalid_cells":int((~valid).sum())})
            item["status"]="LABELS_VALIDATED_SOURCE_PROVENANCE_ONLY"
        except Exception as exc:item["status"]="FAILED_QC";item["qc_error"]=repr(exc)
    manifest["records"].append(item)
    save_manifest()
    print(dstr,item["status"],"class4",sum(x["classes"][4] for x in item.get("scenes",[])),flush=True)
manifest["summary"]={"days_completed":sum(x["status"]=="LABELS_VALIDATED_SOURCE_PROVENANCE_ONLY" for x in manifest["records"]),"scenes":sum(len(x.get("scenes",[])) for x in manifest["records"]),"paired_scenes":0,"trainable_pairs":0}
save_manifest()
print("IMERG_ONLY_SUMMARY",json.dumps(manifest["summary"]),flush=True)
