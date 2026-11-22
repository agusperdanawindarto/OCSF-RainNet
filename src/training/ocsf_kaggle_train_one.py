"""One reproducible training run for OCSF-RainNet (Kaggle GPU only).
Six task-matched models x three fixed seeds, one run per notebook.
Source input: private, calibrated Himawari-9 AHI B13 + IMERG V07B labels.
Never train from the old map-rendered JPEG pilot; never evaluate test during fitting.
Outputs are written to /kaggle/working for validated desktop artifact sync.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS","1")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["PYTHONHASHSEED"]=str(RUN_SEED)
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib, json, csv, datetime, random, math, time, traceback, sys, gc

assert RUN_VARIANT in ("plain","fusion","ordinal","proposed","unet","raunet")
assert RUN_SEED in (42,123,2025)
np.random.seed(RUN_SEED); random.seed(RUN_SEED); tf.random.set_seed(RUN_SEED)
tf.keras.backend.clear_session()
OUT=Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
JOB=f"{RUN_VARIANT}_s{RUN_SEED}"
FILE_PREFIX=f"OCSF_{JOB}"
TIME=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
GPUS=tf.config.list_physical_devices("GPU")
assert GPUS,"Kaggle GPU is required: no silent CPU fallback"
try:
    tf.config.set_visible_devices(GPUS[:1],"GPU")
    tf.config.experimental.set_memory_growth(GPUS[0],True)
except RuntimeError:
    pass

INPUT_ROOT=Path("/kaggle/input")
zips=list(INPUT_ROOT.rglob("paired_127_lossless.zip"))
ZIP=zips[0] if zips else None
exploded=next((p for p in INPUT_ROOT.rglob("paired_127_lossless") if p.is_dir() and len(list(p.glob("*.npz")))==127),None)
if ZIP is None and exploded is None:
    raise RuntimeError("Private calibrated dataset absent: zip and Kaggle-unzipped scenes not found; do not use pilot JPEG")

source=next(iter(INPUT_ROOT.rglob("source_checksum_manifest.json")),None)
splits=next(iter(INPUT_ROOT.rglob("split_by_UTC_date.csv")),None)
model_script=next(iter(INPUT_ROOT.rglob("rainnet_intensity_models.py")),None)
assert source is not None and splits is not None and model_script is not None
provenance=json.loads(source.read_text(encoding="utf8"))
assert len(provenance["images"])==127
if ZIP is not None:
    assert hashlib.sha256(ZIP.read_bytes()).hexdigest()==provenance["zip_sha256"]

sys.path.insert(0,str(model_script.parent))
import rainnet_intensity_models as net
net.CHECKPOINT=next(iter(INPUT_ROOT.rglob("rainnet_binary_best_05.h5")))
net.ARCH=model_script.parent
assert net.pretrained_sha256()==net.ORIGINAL_CHECKPOINT_SHA256
date_group={}
with splits.open(newline="",encoding="utf-8-sig") as f:
    for rec in csv.DictReader(f):
        date_group[rec["utc_date"]]=rec["assigned_split"]
assert len(date_group)==32 and set(date_group.values())=={"train","validation","test"}
assert [sum(1 for x in date_group.values() if x==s) for s in ("train","validation","test")]==[22,5,5]
dataset={}
if ZIP is not None:
    z=ZipFile(ZIP,"r")
    assert len([n for n in z.namelist() if n.endswith(".npz")])==127
else:
    z=None
    assert len(list(exploded.glob("*.npz")))==127
try:
    for record in provenance["images"]:
        raw=(z.read(record["scene"]) if z is not None else (exploded/record["scene"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==record["portable_npz_sha256"]
        with np.load(BytesIO(raw),allow_pickle=False) as d:
            X=np.asarray(d["x_grayscale_float32"],np.float32)
            Y=np.asarray(d["y"],np.uint8)
        assert X.shape==Y.shape==(185,460) and np.isfinite(X).all()
        assert 0<=float(X.min()) and float(X.max())<=1
        labels=np.bincount(Y[Y!=255],minlength=5)
        assert labels.tolist()==record["classes"],(record["key"],labels.tolist(),record["classes"])
        dataset[record["key"]]={"x":X,"y":Y,"date":record["date_utc"],"split":date_group[record["date_utc"]]}
finally:
    if z is not None:z.close()
assert len(dataset)==127
GROUPS={s:sorted(k for k,v in dataset.items() if v["split"]==s) for s in ("train","validation","test")}
assert [len(GROUPS[s]) for s in ("train","validation","test")]==[88,19,20]
train_labels=np.zeros(5,dtype=np.int64)
for key in GROUPS["train"]:
    y=dataset[key]["y"]; train_labels+=np.bincount(y[y!=255],minlength=5)
weights=net.empirical_train_class_weights(train_labels)
loss_fn=net.masked_weighted_ordinal_loss(weights,0.12)
EXPECTED_PARAMS={"plain":142637,"fusion":148234,"ordinal":142632,"proposed":148217,"unet":485853,"raunet":508249}
RUN_CFG={
    "run_id":JOB,"variant":RUN_VARIANT,"seed":RUN_SEED,
    "status":"TRAINING_NOT_YET_EVALUATED","start_utc":TIME(),
    "source":"NOAA/JMA calibrated Himawari-9 AHI B13 matched to GPM IMERG Final V07B",
    "data_identifier":"agusperdana/ocsf-rainnet-calibrated-2024",
    "dataset_archive_sha256":provenance["zip_sha256"],
    "pretrained_rainnet_sha256":net.ORIGINAL_CHECKPOINT_SHA256,
    "input":"float32 grayscale derived from calibrated Kelvin, three identical channels",
    "class_labels":"C0 [0,1), C1 [1,5), C2 [5,10), C3 [10,20], C4 (20,infinity) mm/h; 255 ignored",
    "split_rule":"UTC date disjoint before crop; splits frozen in dataset",
    "train_val_test_count":{k:len(v) for k,v in GROUPS.items()},
    "date_count":{k:len(set(dataset[x]["date"] for x in v)) for k,v in GROUPS.items()},
    "train_class_counts":train_labels.tolist(),"train_class_weights":weights,
    "max_epochs":25,"batch_size":2,"crops_per_train_scene_per_epoch":4,
    "crop_shape":[128,128],"learning_rates":{"warmup":0.001,"finetune":0.0001},
    "frozen_pretrained_epochs":5 if RUN_VARIANT not in ("unet","raunet") else 0,
    "early_stopping_patience":4,
    "loss":"train-frequency sqrt inverse capped 12 and normalized x masked CE +0.12 ordinal severity L1",
    "validation":"all 19 complete val scenes padded to multiple of 16, never test",
    "test":"20 complete independent-date scenes only after best val checkpoint",
    "dataset_admissibility":"radiometric/timestamp QC passed, broader source-independence caveats remain",
    "software":{"tensorflow":tf.__version__,"python":sys.version.split()[0]},
    "gpu_devices":[g.name for g in GPUS]
}
def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf8")
save_json(OUT/f"{FILE_PREFIX}_run_manifest.json",RUN_CFG)

def construct():
    if RUN_VARIANT=="unet":return net.build_unet_baseline(False)
    if RUN_VARIANT=="raunet":return net.build_unet_baseline(True)
    return net.build_rainnet_variant(RUN_VARIANT,pretrained=True,freeze_backbone=True)

model=construct()
assert model.count_params()==EXPECTED_PARAMS[RUN_VARIANT],(model.count_params(),EXPECTED_PARAMS[RUN_VARIANT])
RUN_CFG["parameters"]=model.count_params()
save_json(OUT/f"{FILE_PREFIX}_run_manifest.json",RUN_CFG)
delayed_pool={}
def rgb(a):
    return np.repeat(a[...,None],3,axis=2)
def full(key):
    d=dataset[key]
    x=rgb(d["x"]); y=d["y"]
    xp=np.pad(x,((0,7),(0,4),(0,0)),constant_values=0)
    yp=np.pad(y,((0,7),(0,4)),constant_values=255)
    return xp[None].astype(np.float32),yp[None].astype(np.uint8)
def crop_batches(epoch):
    rng=np.random.default_rng(RUN_SEED*100003+epoch*7919)
    keys=GROUPS["train"].copy()
    rng.shuffle(keys)
    patches=[]
    for key in keys:
        x=dataset[key]["x"]; y=dataset[key]["y"]
        for j in range(4):
            if rng.random()<0.55:
                if key not in delayed_pool:
                    delayed_pool[key]=np.argwhere((y==3)|(y==4))
                points=delayed_pool[key]
            else:
                points=np.empty((0,2),dtype=np.int16)
            if len(points):
                pos=points[int(rng.integers(0,len(points)))]
                r0=int(np.clip(int(pos[0])-int(rng.integers(0,128)),0,185-128))
                c0=int(np.clip(int(pos[1])-int(rng.integers(0,128)),0,460-128))
            else:
                r0=int(rng.integers(0,185-128+1))
                c0=int(rng.integers(0,460-128+1))
            patches.append((key,r0,c0))
    rng.shuffle(patches)
    assert len(patches)==352
    for i in range(0,len(patches),2):
        xs=[];ys=[]
        for key,r,c in patches[i:i+2]:
            xs.append(rgb(dataset[key]["x"][r:r+128,c:c+128]))
            ys.append(dataset[key]["y"][r:r+128,c:c+128])
        yield tf.convert_to_tensor(np.stack(xs),tf.float32),tf.convert_to_tensor(np.stack(ys).astype(np.int32))

def validation_loss():
    losses=[]
    for k in GROUPS["validation"]:
        xx,yy=full(k)
        pp=model(tf.convert_to_tensor(xx),training=False)
        losses.append(float(loss_fn(tf.convert_to_tensor(yy.astype(np.int32)),pp).numpy()))
    return float(np.mean(losses))

def set_phase(epoch):
    warmup=RUN_VARIANT not in ("unet","raunet") and epoch<=5
    if RUN_VARIANT not in ("unet","raunet"):
        for layer in model.rainnet_backbone_layers:
            layer.trainable=not warmup
    return "warmup" if (epoch<=5) else "finetune"

def optimizer_for_phase(phase):
    return tf.keras.optimizers.Adam(learning_rate=1e-3 if phase=="warmup" else 1e-4)

def compile_step(opt):
    @tf.function(reduce_retracing=True)
    def train_step(x,y):
        with tf.GradientTape() as tape:
            probs=model(x,training=True)
            loss=loss_fn(y,probs)
        var=model.trainable_variables
        grad=tape.gradient(loss,var)
        pairs=[(g,v) for g,v in zip(grad,var) if g is not None]
        opt.apply_gradients(pairs)
        return loss
    return train_step

HIST=[]
BEST=float("inf")
WAIT=0
BEST_EPOCH=0
best_path=OUT/f"{FILE_PREFIX}_best.weights.h5"
last_path=OUT/f"{FILE_PREFIX}_last.weights.h5"
history_csv=OUT/f"{FILE_PREFIX}_epoch_history.csv"
last_phase=None
train_step=None
tf.config.optimizer.set_jit(False)
start_time=time.perf_counter()
try:
    for epoch in range(1,26):
        phase=set_phase(epoch)
        if phase!=last_phase:
            optimizer=optimizer_for_phase(phase)
            train_step=compile_step(optimizer)
            last_phase=phase
            if epoch==6:
                WAIT=0
        epoch_started=time.perf_counter()
        train_losses=[]
        for xb,yb in crop_batches(epoch):
            val=float(train_step(xb,yb).numpy())
            if not np.isfinite(val):
                raise FloatingPointError("Nonfinite training loss "+str((epoch,val)))
            train_losses.append(val)
        vloss=validation_loss()
        if not np.isfinite(vloss):raise FloatingPointError("Nonfinite validation loss")
        model.save_weights(str(last_path))
        improvement=vloss < BEST-1e-5
        if improvement:
            BEST=vloss;BEST_EPOCH=epoch;WAIT=0
            model.save_weights(str(best_path))
        elif epoch>5:
            WAIT+=1
        rec={
            "epoch":epoch,"phase":phase,"train_loss":float(np.mean(train_losses)),
            "val_loss":vloss,"best_val_loss":BEST,"improved":improvement,
            "wait":WAIT,"epoch_seconds":time.perf_counter()-epoch_started
        }
        HIST.append(rec)
        with history_csv.open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(HIST[0]));w.writeheader();w.writerows(HIST)
        RUN_CFG["last_epoch"]=epoch
        RUN_CFG["best_epoch"]=BEST_EPOCH
        RUN_CFG["best_validation_loss"]=BEST
        RUN_CFG["status"]="TRAINING"
        RUN_CFG["updated_utc"]=TIME()
        save_json(OUT/f"{FILE_PREFIX}_run_manifest.json",RUN_CFG)
        print("EPOCH",JOB,epoch,phase,"train",round(rec["train_loss"],6),"val",round(vloss,6),"best",round(BEST,6),"wait",WAIT,flush=True)
        if epoch>5 and WAIT>=4:
            print("EARLY_STOP",JOB,epoch,flush=True)
            break
    assert BEST_EPOCH>0 and best_path.exists()
    model.load_weights(str(best_path))
except BaseException as ex:
    RUN_CFG["status"]="ERROR_BEFORE_TEST"
    RUN_CFG["error"]=repr(ex)
    RUN_CFG["traceback"]=traceback.format_exc()[-4000:]
    save_json(OUT/f"{FILE_PREFIX}_run_manifest.json",RUN_CFG)
    raise

def cm_counts(yy,pp):
    yy=np.asarray(yy,np.uint8)
    pp=np.asarray(pp,np.uint8)
    mask=(yy!=255)
    return np.bincount((yy[mask].astype(np.int64)*5+pp[mask].astype(np.int64)),minlength=25).reshape(5,5)
def metric(cm):
    tp=np.diag(cm).astype(float)
    fn=cm.sum(axis=1).astype(float)-tp
    fp=cm.sum(axis=0).astype(float)-tp
    iou=np.divide(tp,tp+fn+fp,out=np.full(5,np.nan),where=(tp+fn+fp)>0)
    dice=np.divide(2*tp,2*tp+fn+fp,out=np.full(5,np.nan),where=(2*tp+fn+fp)>0)
    micro_accuracy=float(tp.sum()/cm.sum()) if cm.sum()>0 else None
    sev_tp=float(cm[3:5,3:5].sum())
    sev_fn=float(cm[3:5,:3].sum())
    sev_fp=float(cm[:3,3:5].sum())
    return {"iou":[float(x) if np.isfinite(x) else None for x in iou],
            "dice":[float(x) if np.isfinite(x) else None for x in dice],
            "macro_iou":float(np.nanmean(iou)),
            "macro_dice":float(np.nanmean(dice)),
            "pixel_accuracy":micro_accuracy,
            "severe_csi":float(sev_tp/(sev_tp+sev_fn+sev_fp)) if (sev_tp+sev_fn+sev_fp)>0 else None,
            "severe_pod":float(sev_tp/(sev_tp+sev_fn)) if (sev_tp+sev_fn)>0 else None,
            "severe_far":float(sev_fp/(sev_tp+sev_fp)) if (sev_tp+sev_fp)>0 else None,
            "severe_tp":int(sev_tp),"severe_fn":int(sev_fn),"severe_fp":int(sev_fp)}

combined=np.zeros((5,5),np.int64)
per_scene=[]
predict_zip=OUT/f"{FILE_PREFIX}_test_predictions.zip"
with ZipFile(predict_zip,"w") as z:
    for key in GROUPS["test"]:
        xx,yy=full(key)
        probs=model(tf.convert_to_tensor(xx),training=False).numpy()[0,:185,:460]
        assert probs.shape==(185,460,5) and np.isfinite(probs).all()
        pred=probs.argmax(-1).astype(np.uint8)
        real=yy[0,:185,:460].copy()
        count=cm_counts(real,pred)
        combined+=count
        result={"key":key,"date":dataset[key]["date"],"classes_actual":np.bincount(real[real!=255],minlength=5).tolist(),"confusion_matrix":count.tolist(),**metric(count)}
        per_scene.append(result)
        b=BytesIO()
        np.savez_compressed(b,actual_C0_C4_uint8=real,predicted_C0_C4_uint8=pred,probabilities_float16=probs.astype(np.float16))
        z.writestr(key+"_test_prediction.npz",b.getvalue())
assert len(per_scene)==20
with (OUT/f"{FILE_PREFIX}_test_confusion_matrix.csv").open("w",newline="",encoding="utf8") as f:
    wr=csv.writer(f);wr.writerow(["actual/predicted","C0","C1","C2","C3","C4"])
    for i,row in enumerate(combined):wr.writerow([f"C{i}",*row])
with (OUT/f"{FILE_PREFIX}_test_per_scene_metrics.csv").open("w",newline="",encoding="utf8") as f:
    keys=["key","date","macro_iou","macro_dice","pixel_accuracy","severe_csi","severe_pod","severe_far","severe_tp","severe_fn","severe_fp"]
    wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
    for sc in per_scene:wr.writerow({k:sc.get(k) for k in keys})
results={"project":"OCSF-RainNet","run":JOB,"model":RUN_VARIANT,"seed":RUN_SEED,"status":"COMPLETE",
         "data_source":"Calibrated AHI B13 / satellite IMERG V07B, NOT independent BMKG observations",
         "train_scene_count":88,"validation_scene_count":19,"test_scene_count":20,
         "test_UTC_dates":sorted(set(dataset[k]["date"] for k in GROUPS["test"])),
         "epoch_history":HIST,"epochs_executed":len(HIST),"best_epoch":BEST_EPOCH,
         "best_validation_loss":BEST,"parameters":model.count_params(),
         "total_training_seconds":time.perf_counter()-start_time,
         "aggregate_test_confusion_matrix":combined.tolist(),
         "aggregate_test_metrics":metric(combined),
         "per_scene":per_scene,"finished_utc":TIME(),
         "rainnet_checkpoint_sha256":net.ORIGINAL_CHECKPOINT_SHA256,
         "data_archive_sha256":provenance["zip_sha256"]}
save_json(OUT/f"{FILE_PREFIX}_test_results.json",results)
fig,ax=plt.subplots(figsize=(7,6))
im=ax.imshow(np.log1p(combined),cmap="Blues")
ax.set_xticks(range(5),[f"C{i}" for i in range(5)])
ax.set_yticks(range(5),[f"C{i}" for i in range(5)])
ax.set_xlabel("Predicted rainfall class")
ax.set_ylabel("Actual IMERG rainfall class")
ax.set_title("OCSF-RainNet held-out confusion matrix: "+JOB)
for i in range(5):
    for j in range(5):
        ax.text(j,i,f"{combined[i,j]:,}",ha="center",va="center",fontsize=8,color="white" if np.log1p(combined[i,j])>0.6*np.log1p(combined.max()) else "black")
fig.colorbar(im,ax=ax,label="log(1 + cell count)")
fig.tight_layout()
fig.savefig(OUT/f"{FILE_PREFIX}_confusion_600dpi.png",dpi=600,bbox_inches="tight")
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
ax.plot([h["epoch"] for h in HIST],[h["train_loss"] for h in HIST],label="Train (crop)")
ax.plot([h["epoch"] for h in HIST],[h["val_loss"] for h in HIST],label="Validation (full scene)")
ax.axvline(BEST_EPOCH,color="gray",linestyle="--",label="Best epoch")
ax.set_xlabel("Epoch");ax.set_ylabel("Masked weighted loss")
ax.set_title("OCSF-RainNet learning curves: "+JOB);ax.legend()
fig.tight_layout()
fig.savefig(OUT/f"{FILE_PREFIX}_learning_curves_600dpi.png",dpi=600,bbox_inches="tight")
plt.close(fig)
RUN_CFG.update({"status":"COMPLETE","finished_utc":TIME(),"best_epoch":BEST_EPOCH,
    "test_macro_iou":results["aggregate_test_metrics"]["macro_iou"],
    "test_macro_dice":results["aggregate_test_metrics"]["macro_dice"],
    "test_severe_csi":results["aggregate_test_metrics"]["severe_csi"]})
save_json(OUT/f"{FILE_PREFIX}_run_manifest.json",RUN_CFG)
(OUT/f"{FILE_PREFIX}_RUN_COMPLETE.flag").write_text(TIME(),encoding="utf8")
print("RUN_COMPLETE",JOB,results["aggregate_test_metrics"],"epochs",len(HIST),flush=True)
