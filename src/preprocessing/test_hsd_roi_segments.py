"""Safely identify subset HSD segments for the Indonesia ROI by exact comparison to all-10 full-disk pilot."""
from satpy import Scene
from pyresample import create_area_def
from pathlib import Path
import numpy as np,json,traceback
import dask
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
src=R/"raw/himawari/noaa_ahi_l1b/2024_01_07/1410/B13"
ref=np.load(R/"processed/ahi_calibrated_2024/20240107_1410_Himawari9_B13_AHI_BT_IMERG_grid.npz")["brightness_temperature_K"]
allfiles=sorted(src.glob("*.DAT.bz2"))
area=create_area_def("IMERG_ROI",{"proj":"longlat","datum":"WGS84"},width=460,height=185,area_extent=(95.0,-11.5,141.0,7.0),units="degrees")
out={"reference":{"segments":10,"shape":list(ref.shape)},"tests":[]}
for indices in ((4,5,6,7,8),(5,6,7),(4,5,6,7),(3,4,5,6,7,8)):
    a={"segments":list(indices)}
    files=[str(f) for f in allfiles if any(f"_S{s:02d}10" in f.name for s in indices)]
    try:
        with dask.config.set(scheduler="threads",num_workers=2):
            scene=Scene(reader="ahi_hsd",filenames=files)
            scene.load(["B13"])
            sub=scene.resample(area,resampler="nearest")
            img=np.asarray(sub["B13"].compute().values,dtype=np.float32)
        a["shape"]=list(img.shape);a["valid_fraction"]=float(np.isfinite(img).mean())
        valid=np.isfinite(img)&np.isfinite(ref)
        a["max_abs_diff_K"]=float(np.max(np.abs(img[valid]-ref[valid]))) if valid.any() else None
        a["equal_to_full_scene"]=bool(valid.all() and np.max(np.abs(img[valid]-ref[valid]))<1e-3)
    except Exception as exc:
        a["error"]=repr(exc)
        a["traceback"]=traceback.format_exc()[-1500:]
    out["tests"].append(a)
    print("SEGMENTS_TEST",json.dumps({k:v for k,v in a.items() if k!="traceback"}),flush=True)
p=R/"experiments/2026-11-21_kaggle_agusperdana/hsd_minimum_segments_validation.json"
p.write_text(json.dumps(out,indent=2),encoding="utf8")
