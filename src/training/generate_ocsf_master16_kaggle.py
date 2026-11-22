"""Create a self-contained multi-run Kaggle GPU notebook to complete 16 remaining OCSF runs
even if the Windows laptop lid is closed. Assumes first two model seeds are separate Kaggle kernels.
"""
from pathlib import Path
import json,hashlib
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
src=(R/"scripts/ocsf_kaggle_train_one.py").read_text(encoding="utf8")
plan=json.loads((R/"kaggle/train_runs/training_manifest.json").read_text(encoding="utf8"))["runs"]
skip={("plain",42),("fusion",42)}
tasks=[(r["variant"],r["seed"]) for r in plan if (r["variant"],r["seed"]) not in skip]
assert len(tasks)==16 and tasks[0]==("ordinal",42)
init="import json,datetime,traceback,gc,os\nfrom pathlib import Path\n"\
     +"_source="+repr(src)+"\n"\
     +"_jobs="+repr(tasks)+"\n"\
     +"_out=Path('/kaggle/working')\n"\
     +"_log={'mode':'KAGGLE_REMOTE_16_JOB_MASTER','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}\n"\
     +"for variant,seed in _jobs:\n"\
     +"    print('MASTER_START',variant,seed,flush=True)\n"\
     +"    _status={'variant':variant,'seed':seed,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"\
     +"    try:\n"\
     +"        _scope={'RUN_VARIANT':variant,'RUN_SEED':seed,'__name__':'__main__'}\n"\
     +"        exec(compile(_source,'ocsf_kaggle_train_one.py','exec'),_scope)\n"\
     +"        _status['status']='COMPLETE'\n"\
     +"    except BaseException as error:\n"\
     +"        _status['status']='ERROR'\n"\
     +"        _status['error']=repr(error)\n"\
     +"        _status['traceback']=traceback.format_exc()[-5000:]\n"\
     +"        print('MASTER_FAILED',variant,seed,repr(error),flush=True)\n"\
     +"    finally:\n"\
     +"        _status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n"\
     +"        _log['jobs'].append(_status)\n"\
     +"        (_out/'OCSF_16_Run_Master_Status.json').write_text(json.dumps(_log,indent=2),encoding='utf8')\n"\
     +"        gc.collect()\n"\
     +"_log['completed_count']=sum(x['status']=='COMPLETE' for x in _log['jobs'])\n"\
     +"_log['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n"\
     +"(_out/'OCSF_16_Run_Master_Status.json').write_text(json.dumps(_log,indent=2),encoding='utf8')\n"\
     +"print('MASTER_COMPLETED',_log['completed_count'],'OF',len(_jobs),flush=True)\n"\
     +"assert _log['completed_count']==len(_jobs),'At least one job failed; inspect OCSF_16_Run_Master_Status.json'\n"
nb={"cells":[
    {"cell_type":"markdown","metadata":{},"source":["# OCSF-RainNet — Kaggle-only 16-run continuation\n",
         "Runs all remaining OCSF RainNet variants/seeds in the remote Kaggle GPU without depending on a Windows-local dispatcher. Separate Kaggle jobs cover plain seed42 and fusion seed42. This kernel writes all per-run artifacts and a master completion manifest.\n"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in init.splitlines()]}
],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}
d=R/"kaggle/train_runs/ocsf-rainnet-master16-20261121"
d.mkdir(parents=True,exist_ok=True)
(d/"OCSF_Master_16_Training.ipynb").write_text(json.dumps(nb,ensure_ascii=False),encoding="utf8")
meta={"id":"agusperdana/ocsf-rainnet-master16-calibrated-2024","title":"OCSF RainNet Master16 Calibrated 2024","code_file":"OCSF_Master_16_Training.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,"dataset_sources":["agusperdana/ocsf-rainnet-calibrated-2024"],"competition_sources":[],"kernel_sources":[]}
(d/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
compile(init,str(d/"OCSF_Master_16_Training.ipynb"),"exec")
print("MASTER_READY",d,"runs",len(tasks),"sha",hashlib.sha256(init.encode()).hexdigest(),flush=True)
