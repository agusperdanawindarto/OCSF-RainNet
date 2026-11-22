"""OCSF-RainNet qualitative held-out evaluation using archived EXACT 2024 test predictions.
Read-only Kaggle ZIPs; 3 deliberately chosen contrasting cases from seed 42.
No inference/training. Every spatial prediction loaded from original per-run NPZ.
"""
from pathlib import Path
import io, zipfile, json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
E=R/"experiments/OCSF_2024_All_18_Train_Runs"
DEST=R/"PAPER SPRINGER/QUALITATIVE_RESULTS_2026"
DEST.mkdir(exist_ok=True)
palette=["#d5dade","#3b8bdb","#2ba58c","#f4b338","#cb3340"]
labels=["C0 <1","C1 1-<5","C2 5-<10","C3 10-20","C4 >20"]
cmap=ListedColormap(palette)
norm=BoundaryNorm(np.arange(-.5,5.5),cmap.N)
miss_color="#ececec"

CASES=[
 {"key":"20241211_2010","top":0,"left":75,"kind":"localized positive severe-rain detection","abbr":"gain"},
 {"key":"20240603_1410","top":42,"left":209,"kind":"missed severe rain across all four variants","abbr":"failure"},
 {"key":"20240811_0210","top":0,"left":77,"kind":"fusion-only outperforming the proposed model","abbr":"counterexample"}
]
H,W=64,96
models=["plain","fusion","ordinal","proposed"]
names={"plain":"Multiclass RainNet","fusion":"Fusion-only","ordinal":"Ordinal-only","proposed":"OCSF-RainNet"}
report={"source":"Actual Kaggle seed-42 held-out NPZ predictions, not regenerated", "published_for":"post-hoc qualitative diagnostic cases (not unbiased sample)", "rain_groups":"severe = C3 or C4", "cases":[]}
roi_rows=[]
full_rows=[]
data_by_key={}

def get_data(key):
    arrays={}
    for m in models:
        directory=E/f"ocsf-rainnet-fit-{m}-seed-42" if m in ("plain","fusion") else E/"master16"
        zpath=directory/f"OCSF_{m}_s42_test_predictions.zip"
        with zipfile.ZipFile(zpath) as archive:
            with np.load(io.BytesIO(archive.read(key+"_test_prediction.npz")),allow_pickle=False) as src:
                yy=np.asarray(src["actual_C0_C4_uint8"],dtype=np.uint8)
                pp=np.asarray(src["predicted_C0_C4_uint8"],dtype=np.uint8)
        if "target" not in arrays: arrays["target"]=yy
        else: assert np.array_equal(arrays["target"],yy)
        arrays[m]=pp
        # Cross-check against FULL-scene original 5x5 test confusion matrix.
        run=json.loads((directory/f"OCSF_{m}_s42_test_results.json").read_text(encoding="utf8"))
        scene=next(x for x in run["per_scene"] if x["key"]==key)
        assert yy.shape==pp.shape==(185,460)
        ct=np.zeros((5,5),dtype=np.int64)
        v=(yy<5)&(pp<5)
        np.add.at(ct,(yy[v],pp[v]),1)
        assert ct.tolist()==scene["confusion_matrix"], (m,key)
        full_rows.append({"scene":key,"model":m,"seed":42,"severe_actual":int(((yy==3)|(yy==4)).sum()),
                        "C4_actual":int((yy==4).sum()),
                        "TP":scene["severe_tp"],"FP":scene["severe_fp"],"FN":scene["severe_fn"],
                        "CSI":scene["severe_csi"],"macro_IoU":scene["macro_iou"]})
    inp=R/f"processed/aligned_multievent_2024/{key}_B13_BT_IMERG_V07B.npz"
    with np.load(inp,allow_pickle=False) as source:
        arrays["BT_K"]=np.asarray(source["brightness_temperature_K"],dtype=np.float32)
        assert np.array_equal(arrays["target"],source["y"])
    arrays["input_sha256"]=hashlib.sha256(inp.read_bytes()).hexdigest()
    return arrays

def add_categorical(ax,y,title):
    v=np.ma.masked_where(y==255,y)
    ax.imshow(v,cmap=cmap,norm=norm,interpolation="nearest",origin="upper",aspect="equal")
    ax.set_title(title,fontsize=11,pad=5,fontweight="semibold")
    ax.set_xlabel("Grid column within selected ROI",fontsize=8)
    ax.set_ylabel("Grid row within selected ROI",fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks([0,W-1]);ax.set_yticks([0,H-1])

def region_summary(y,p):
    v=(y<5)&(p<5)
    gt=((y>=3)&(y<=4))&v
    pd=((p>=3)&(p<=4))&v
    tp=int((gt&pd).sum());fp=int(((~gt)&pd&v).sum());fn=int((gt&(~pd)).sum())
    c4_gt=int(((y==4)&v).sum());c4_tp=int(((y==4)&(p==4)&v).sum())
    return dict(TP=tp,FP=fp,FN=fn,CSI=(tp/(tp+fp+fn) if tp+fp+fn else None),C4_ground_truth=c4_gt,C4_correct=c4_tp)

for c in CASES:
    key=c["key"]
    data=get_data(key);data_by_key[key]=data
    top,left=c["top"],c["left"]
    roi={k:(v[top:top+H,left:left+W] if isinstance(v,np.ndarray) and v.shape==(185,460) else v) for k,v in data.items()}
    c["ROI"]=f"rows {top}-{top+H-1}, columns {left}-{left+W-1}"
    report["cases"].append({"scene":key,"UTC_scene_time":f"{key[0:4]}-{key[4:6]}-{key[6:8]}T{key[9:11]}:{key[11:13]}:00Z",
                            "selected_for":c["kind"],"ROI":c["ROI"],"source_NPZ_SHA256":data["input_sha256"]})
    fig,axs=plt.subplots(2,3,figsize=(13.0,8.45),sharex=True,sharey=True)
    axs=axs.ravel()
    axs[0].imshow(roi["BT_K"],cmap="gray_r",vmin=180,vmax=320,interpolation="nearest",origin="upper",aspect="equal")
    axs[0].set_title("(a) Calibrated AHI B13 BT (K)",fontsize=11,pad=5,fontweight="semibold")
    for loc,lab in [(1,"(b) IMERG V07B target"),(2,"(c) RainNet"),(3,"(d) Fusion-only"),(4,"(e) Ordinal-only"),(5,"(f) OCSF-RainNet")]:
        name={1:"target",2:"plain",3:"fusion",4:"ordinal",5:"proposed"}[loc]
        add_categorical(axs[loc],roi[name],lab)
    for ax in axs:
        ax.set_xticks([0,W-1]);ax.set_yticks([0,H-1])
        ax.tick_params(labelsize=8)
        if not ax.get_xlabel():ax.set_xlabel("ROI grid column",fontsize=8)
        if not ax.get_ylabel():ax.set_ylabel("ROI grid row",fontsize=8)
    fig.legend(handles=[Patch(facecolor=palette[k],label=labels[k]) for k in range(5)],loc="lower center",
               ncol=5,frameon=False,bbox_to_anchor=(.5,.055),fontsize=10)
    fig.suptitle(f"Actual held-out segmentations: {key} UTC | {c['kind']}\n{c['ROI']}, seed 42",fontsize=13,fontweight="bold",y=.965)
    fig.text(.5,.018,"Reference is satellite-estimated IMERG, not independent ground rainfall. Crops are post-hoc diagnostic choices.",
             ha="center",fontsize=8.5)
    fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.12,hspace=.24,wspace=.20)
    out=DEST/f"Figure_{10+CASES.index(c)}_Qualitative_{c['abbr']}.png"
    fig.savefig(out,dpi=600,facecolor="white",bbox_inches="tight",pil_kwargs={"compress_level":8})
    plt.close(fig)
    for m in models:
        q=region_summary(roi["target"],roi[m])
        roi_rows.append({"scene":key,"selection":c["kind"],"seed":42,"ROI_top":top,"ROI_left":left,
                        "ROI_size":"64x96","model":m,**q})
    print("FIG",out.name,out.stat().st_size,flush=True)

# Fig. 13 diagnostic severe-rain TP/FP/FN maps; no synthetic predictions.
err_cmap=ListedColormap(["#f1f1f1","#359b73","#ee9a36","#cd3b4a"])
err_norm=BoundaryNorm(np.arange(-.5,4.5),4)
fig,axs=plt.subplots(3,3,figsize=(11.7,9.5),sharex=True,sharey=True)
for row,c in enumerate(CASES):
    data=data_by_key[c["key"]];top,left=c["top"],c["left"]
    y=data["target"][top:top+H,left:left+W]
    for col,m in enumerate(("plain","ordinal","proposed")):
        p=data[m][top:top+H,left:left+W]
        v=(y<5)&(p<5);gt=(y>=3)&(y<=4)&v;pd=(p>=3)&(p<=4)&v
        classes=np.zeros(y.shape,dtype=np.uint8)
        classes[gt&pd]=1 # severe TP
        classes[(~gt)&pd&v]=2 # severe FP
        classes[gt&(~pd)]=3 # severe FN
        ax=axs[row,col]
        ax.imshow(classes,cmap=err_cmap,norm=err_norm,interpolation="nearest",aspect="equal")
        ax.set_title(f"{c['key']} · {names[m]}",fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
fig.legend([Patch(facecolor=c) for c in err_cmap.colors],
           ["Neither target nor predicted severe","TP: severe recognized","FP: false severe alarm","FN: missed severe target"],
           ncol=2,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False,fontsize=10)
fig.suptitle("Fig. 13: Spatial error modes for true C3/C4 vs predicted C3/C4 (seed 42)",
             fontweight="bold",fontsize=13)
fig.subplots_adjust(left=.04,right=.98,top=.92,bottom=.105,hspace=.18,wspace=.07)
out=DEST/"Figure_13_Severe_Rain_TP_FP_FN.png"
fig.savefig(out,dpi=600,facecolor="white",bbox_inches="tight",pil_kwargs={"compress_level":8});plt.close(fig)
print("FIG",out.name,out.stat().st_size,flush=True)

# Fig. 14 ordinal-distance maps for one matched ROI (true class vs predicted index).
c=CASES[0];data=data_by_key[c["key"]];top,left=c["top"],c["left"]
y=data["target"][top:top+H,left:left+W]
fig,axs=plt.subplots(1,3,figsize=(13,4.8))
for ax,m in zip(axs,("plain","ordinal","proposed")):
    p=data[m][top:top+H,left:left+W]
    err=np.abs(p.astype(np.int16)-y.astype(np.int16))
    err=np.ma.masked_where((y==255)|(p==255),err)
    im=ax.imshow(err,cmap="magma",vmin=0,vmax=4,interpolation="nearest",aspect="equal")
    ax.set_title(names[m],fontsize=11,fontweight="semibold")
    ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axs.tolist(),shrink=.75,pad=.02,ticks=[0,1,2,3,4],label="Absolute ordinal class-distance error")
fig.suptitle(f"Fig. 14: Ordinal severity-distance error on matched test ROI {c['key']} (seed 42)",fontsize=12.5,fontweight="bold")
fig.subplots_adjust(left=.025,right=.93,top=.87,bottom=.11,wspace=.05)
out=DEST/"Figure_14_Ordinal_Distance_Error_Maps.png"
fig.savefig(out,dpi=600,facecolor="white",bbox_inches="tight",pil_kwargs={"compress_level":8});plt.close(fig)
print("FIG",out.name,out.stat().st_size,flush=True)

with (DEST/"Table_11_Qualitative_ROI_Severe_Error.csv").open("w",newline="",encoding="utf8") as f:
    writer=csv.DictWriter(f,fieldnames=list(roi_rows[0].keys()))
    writer.writeheader();writer.writerows(roi_rows)
with (DEST/"Table_12_Matched_Full_Scene_Severe_Metrics.csv").open("w",newline="",encoding="utf8") as f:
    writer=csv.DictWriter(f,fieldnames=list(full_rows[0].keys()))
    writer.writeheader();writer.writerows(full_rows)
report["roi_table"]="Table_11_Qualitative_ROI_Severe_Error.csv"
report["full_scene_table"]="Table_12_Matched_Full_Scene_Severe_Metrics.csv"
(DEST/"Figure_10_14_Provenance_and_Selection.json").write_text(json.dumps(report,indent=2),encoding="utf8")
print("ALL_QUALITATIVE_ARTIFACTS_PRODUCED",flush=True)
