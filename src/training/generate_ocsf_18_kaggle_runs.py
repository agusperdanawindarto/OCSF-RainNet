"""Generate 18 independent private Kaggle training notebooks from one audited source.
Do not launch them until calibrated 127-scene dataset exists and GPU is available.
"""
from pathlib import Path
import json,datetime,hashlib
ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
SOURCE=ROOT/"scripts/ocsf_kaggle_train_one.py"
T=ROOT/"kaggle/train_runs"
T.mkdir(parents=True,exist_ok=True)
body=SOURCE.read_text(encoding="utf8")
compile("RUN_VARIANT='plain'\nRUN_SEED=42\n"+body,str(SOURCE),"exec")
variants=["plain","fusion","ordinal","proposed","unet","raunet"]
seeds=[42,123,2025]
runs=[]
for seed in seeds:
    for variant in variants:
        slug=f"ocsf-rainnet-fit-{variant}-seed-{seed}"
        d=T/slug;d.mkdir(parents=True,exist_ok=True)
        nb_filename="ocsf_train_one.ipynb"
        title=f"OCSF RainNet fit {variant} seed {seed}"
        assert len(title)<=70
        cell=f"RUN_VARIANT={variant!r}\nRUN_SEED={seed}\n"+body
        nb={"cells":[
            {"cell_type":"markdown","metadata":{},"source":[f"# OCSF-RainNet {variant} / seed {seed}\n", "Calibrated Himawari AHI B13 and NASA GPM IMERG Final V07B. Date-disjoint training/validation/test, 25-epoch maximum, early stop 4. Untrained result and synthetic smoke tests are not part of this notebook.\n"]},
            {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in cell.splitlines()]}
        ],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}
        metadata={"id":"agusperdana/"+slug,"title":title,"code_file":nb_filename,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,"dataset_sources":["agusperdana/ocsf-rainnet-calibrated-2024"],"competition_sources":[],"kernel_sources":[]}
        (d/nb_filename).write_text(json.dumps(nb,ensure_ascii=False),encoding="utf8")
        (d/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
        runs.append({"variant":variant,"seed":seed,"slug":slug,"kaggle_ref":"agusperdana/"+slug,"dir":str(d),"status":"READY_TO_LAUNCH","source_sha256":hashlib.sha256(body.encode("utf8")).hexdigest()})
manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"dataset":"agusperdana/ocsf-rainnet-calibrated-2024","runs":runs,"status":"READY_TO_DISPATCH_AFTER_DATASET_VERIFICATION","gpu_only":True}
(T/"training_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
print("CREATED",len(runs),"PRIVATE_TRAINING_NOTEBOOKS","manifest",T/"training_manifest.json",flush=True)
