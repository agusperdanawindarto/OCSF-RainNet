"""Compare public NOAA ERDDAP IMERG Final V07 with local NASA HDF5-derived pilot, without assuming zeros are nodata."""
import urllib.request, urllib.parse, numpy as np, io, json, datetime
from scipy.io import netcdf_file
from pathlib import Path
root=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
pilot=np.load(root/"processed/aligned/20240107_1400_B13_IMERG_V07.npz",allow_pickle=False)
url="https://erddap.aoml.noaa.gov/hdb/erddap/griddap/precipitation_2024_v07.nc?"+urllib.parse.quote("precipitation[(2024-01-07T14:00:00Z)][(-11.45):1:(6.95)][(95.05):1:(140.95)]",safe="[]():,.-")
report={"url":url,"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    req=urllib.request.Request(url,headers={"User-Agent":"OCSF-RainNet-research-validation/1"})
    with urllib.request.urlopen(req,timeout=90) as response:
        raw=response.read(6_000_000)
        report["http_status"]=response.status
        report["bytes"]=len(raw);report["content_type"]=response.headers.get("Content-Type")
    f=netcdf_file(io.BytesIO(raw),"r",mmap=False)
    report["variables"]={k:list(v.shape) for k,v in f.variables.items()}
    y=np.asarray(f.variables["precipitation"][:],dtype=np.float64).squeeze()
    ref=np.asarray(pilot["rain_rate_mm_hr"],np.float64)
    report["precip_shape"]=list(y.shape)
    report["metadata"]={name:str(getattr(f.variables["precipitation"],name,"")) for name in ("missing_value","_FillValue","units")}
    if y.shape==ref.shape:
        matched=y
    elif y.T.shape==ref.shape:
        matched=y.T
    else:
        raise ValueError("ERDDAP subset shape mismatch: "+str(y.shape))
    candidates={"raw":matched,"flip_lat":np.flipud(matched)}
    report["comparisons"]={}
    for key,arr in candidates.items():
        mask=np.isfinite(arr)&np.isfinite(ref)
        vals=arr[mask];gt=ref[mask]
        report["comparisons"][key]={"n":int(mask.sum()),"raw_nan":int(np.isnan(arr).sum()),"raw_zero":int(np.sum(arr==0)),"pilot_zero":int(np.sum(ref==0)),"mae":float(np.mean(np.abs(vals-gt))),"max_abs_diff":float(np.max(np.abs(vals-gt))),"close_1e-3_fraction":float(np.mean(np.abs(vals-gt)<=1e-3))}
    f.close()
except Exception as e:
    report["error"]=repr(e)
path=root/"experiments/2026-11-21_kaggle_agusperdana/erddap_nasa_pilot_validation.json"
path.write_text(json.dumps(report,indent=2),encoding="utf8")
print(json.dumps(report,indent=2)[:11000])
