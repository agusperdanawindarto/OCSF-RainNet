"""Author-edited, data-driven figures for the OCSF-RainNet reproducibility repository.
No GPU computation. Source = verified 18-run CSV and published experimental methodology.
"""
from pathlib import Path
import csv,numpy as np,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import statistics
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN\github_release\OCSF-RainNet")
RESULT=R/"figures/results"; METHODS=R/"figures/methods"
for p in (RESULT,METHODS):p.mkdir(parents=True,exist_ok=True)
cols={"blue":"#cce4f6","deep":"#1a4975","gold":"#ffdf85","olive":"#55878a","purple":"#e8e5f6","dark":"#1b2940","border":"#3b618b","green":"#d0eddf","white":"#ffffff","warn":"#fff0e9"}
model_order=["plain","fusion","ordinal","proposed","unet","raunet"]
display={"plain":"RainNet","fusion":"Fusion-only","ordinal":"Ordinal-only","proposed":"OCSF-RainNet","unet":"U-Net","raunet":"RA-UNet-style"}
data=list(csv.DictReader((R/"tables/Table_05_Mean_SD_6_Models.csv").open(encoding="utf8")))
summary={x["model"]:x for x in data}
seeds=list(csv.DictReader((R/"tables/Table_06_All_18_Runs.csv").open(encoding="utf8")))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"savefig.facecolor":"white"})

def save(fig,path):
    fig.savefig(str(path)+".png",dpi=350,bbox_inches="tight",facecolor="white")
    fig.savefig(str(path)+".svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)

def box(ax,x,y,w,h,title,detail="",color=None,fs=11):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.16",
            facecolor=(color or cols["blue"]),edgecolor=cols["border"],linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*.66,title,ha="center",va="center",fontsize=fs,fontweight="bold",color=cols["dark"],wrap=True)
    if detail:
        ax.text(x+w/2,y+h*.30,detail,ha="center",va="center",fontsize=max(7.2,fs-2.5),color=cols["dark"],wrap=True)
def arrow(ax,x1,y1,x2,y2,col=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,linewidth=1.5,color=col or cols["deep"],connectionstyle="arc3,rad=0"))
def banner(ax,text,y,size=15):
    ax.text(9,y,text,fontsize=size,weight="bold",ha="center",va="center",color=cols["deep"])

# Figure 2 – actual complete experimental workflow and lineage.
fig,ax=plt.subplots(figsize=(18,9));ax.set_xlim(0,18);ax.set_ylim(0,10);ax.axis("off")
banner(ax,"Figure 2. Research framework and architectural lineage (completed 2024 experiment)",9.62,15)
box(ax,.4,7.55,4.1,1.45,"Stage 1 · Satellite acquisition","Himawari-9 AHI B13 Level-1b HSD\nNASA GPM IMERG Final V07B",cols["blue"],11)
box(ax,5.05,7.55,4.0,1.45,"Stage 2 · Source pairing and QC","Calibrated BT (K), UTC screening,\n0.1° grid registration; 127 pairs",cols["blue"],11)
box(ax,9.62,7.55,3.65,1.45,"Stage 3 · Frozen data partitions","32 UTC dates; 88 train, 19 val,\n20 held-out test scenes",cols["green"],10.5)
box(ax,13.85,7.55,3.7,1.45,"Stage 4 · Matched experiments","6 architectures × 3 seeds\n42 / 123 / 2025",cols["gold"],11)
for a,b in [((4.5,8.27),(5.02,8.27)),((9.05,8.27),(9.59,8.27)),((13.27,8.27),(13.82,8.27))]:arrow(ax,*a,*b)
box(ax,.38,4.95,4.2,1.55,"Published RainNet (2024)","Original pretrained binary-cloud\nresidual encoder–decoder",cols["purple"],11)
box(ax,5.20,5.75,4.7,1.15,"Control: five-class RainNet","Same pretrained backbone + softmax",cols["blue"],10.5)
box(ax,5.2,4.15,4.7,1.15,"Proposed: OCSF-RainNet","Gated cross-scale fusion + ordinal head",cols["gold"],10.5)
box(ax,10.75,5.75,6.65,1.15,"Independent baselines","Task-matched U-Net and RA-UNet-style",cols["green"],10.4)
box(ax,10.75,4.15,6.65,1.15,"Ablations","Fusion-only and ordinal-only components",cols["purple"],10.4)
arrow(ax,4.58,5.73,5.13,6.12);arrow(ax,4.58,5.3,5.13,4.75)
box(ax,3.5,1.7,11.0,1.38,"Stage 5 · Frozen, date-held-out evaluation completed","Full-scene predictions, per-class IoU/Dice, severe CSI/POD/FAR,\nconfusion matrices, seed variability, model size and GPU timing",cols["green"],12)
for x in [7.55,14.0]: arrow(ax,x,4.05,x,3.20)
ax.annotate("",xy=(9,3.1),xytext=(9,3.0),arrowprops={"arrowstyle":"-|>","lw":1.3,"color":cols["deep"]})
ax.text(9,.85,"Limits: IMERG is satellite-estimated supervision; C4 scores remain near zero; weather-event and pretraining-overlap audits pending.",
        ha="center",va="center",fontsize=9.7,color="#9a3d20")
fig.tight_layout()
save(fig,METHODS/"Figure_2_Research_Framework_Updated")

# Figure 3 – same supervisory target; separate model training.
fig,ax=plt.subplots(figsize=(18,8.4));ax.set_xlim(0,18);ax.set_ylim(0,9.4);ax.axis("off")
banner(ax,"Figure 3. OCSF-RainNet architecture and task-matched RainNet control",9.0,15)
box(ax,.4,4.75,3.9,2.35,"Himawari-9 AHI Band 13","Calibrated Level-1b brightness\ntemperature (K); 185×460×3\nfixed grayscale channels",cols["blue"],11.5)
box(ax,4.95,4.85,4.2,2.1,"Pretrained RainNet","Residual encoder 4/8/16/32,\nbridge 64, decoder 32/16/8/4\nshared initial checkpoint",cols["blue"],11.5)
arrow(ax,4.3,5.9,4.94,5.9)
box(ax,10.05,6.35,3.6,1.55,"A · Multiclass control","1×1 conv → 5 logits → softmax",cols["blue"],11)
box(ax,10.05,3.95,3.6,1.75,"B · Context-gated fusion","Three encoder skips + decoder\n32-channel context, gated 16 ch",cols["gold"],11)
arrow(ax,9.15,6.25,10.0,7.05);arrow(ax,9.15,5.42,10,4.75)
box(ax,14.25,6.35,3.25,1.55,"Baseline output","Five probabilities p(C0–C4)",cols["purple"],10.6)
box(ax,14.2,3.95,3.35,1.75,"Ordinal exceedance head","Four q thresholds q1 ≥ q2 ≥ q3 ≥ q4\np0...p4 from adjacent differences",cols["gold"],10)
arrow(ax,13.65,7.1,14.2,7.1);arrow(ax,13.65,4.8,14.15,4.8)
box(ax,5.1,1.3,8.1,1.42,"Shared NASA IMERG Final V07B target","C0 <1; C1 1–<5; C2 5–<10; C3 10–20; C4 >20 mm/h",cols["purple"],11)
ax.annotate("",xy=(15.4,6.32),xytext=(13.13,2.10),arrowprops=dict(arrowstyle="-|>",lw=1.4,linestyle="dashed",color="#7154a2"))
ax.annotate("",xy=(15.4,3.90),xytext=(13.13,2.10),arrowprops=dict(arrowstyle="-|>",lw=1.4,linestyle="dashed",color="#7154a2"))
ax.text(3.9,.45,"Models were trained separately with identical data partitions, labels and training objectives; dashed arrows denote supervision only.",
        ha="left",fontsize=9.3,color=cols["deep"])
fig.tight_layout()
save(fig,METHODS/"Figure_3_Model_Architecture_Updated")

# Figure 5 – mean macro IoU with SD.
fig,ax=plt.subplots(figsize=(10.2,5.2))
x=np.arange(6);y=[float(summary[m]["macro_iou_mean"]) for m in model_order];err=[float(summary[m]["macro_iou_sd"]) for m in model_order]
bars=ax.bar(x,y,yerr=err,capsize=4)
ax.set_xticks(x,[display[m] for m in model_order],rotation=20,ha="right")
ax.set_ylim(0,.35);ax.set_ylabel("Held-out Macro IoU (mean ± SD, n=3 seeds)")
ax.set_title("OCSF-RainNet and same-task controls on five held-out dates")
for b,v in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,v+.007,f"{v:.3f}",ha="center",fontsize=8)
ax.grid(axis="y",alpha=.25);fig.tight_layout();save(fig,RESULT/"Figure_5_Macro_IoU_3_Seeds")

# Figure 6 – Severe rain CSI: C3+C4 vs all else.
fig,ax=plt.subplots(figsize=(10.2,5.2))
y=[float(summary[m]["severe_csi_mean"]) for m in model_order];err=[float(summary[m]["severe_csi_sd"]) for m in model_order]
bars=ax.bar(x,y,yerr=err,capsize=4)
ax.set_xticks(x,[display[m] for m in model_order],rotation=20,ha="right")
ax.set_ylim(0,max(.1,max(a+b for a,b in zip(y,err))*1.15))
ax.set_ylabel("Severe-rain CSI (mean ± SD, n=3)")
ax.set_title("Detection of classes C3–C4: IMERG-supervised test fields")
for b,v in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,v+.002,f"{v:.3f}",ha="center",fontsize=8)
ax.grid(axis="y",alpha=.25);fig.tight_layout();save(fig,RESULT/"Figure_6_Severe_CSI_3_Seeds")

# Figure 7 – seed-wise Macro IoU for all model variants.
fig,ax=plt.subplots(figsize=(10,5.2))
for m in model_order:
    vals=[float(next(t["macro_iou"] for t in seeds if t["model"]==m and int(t["seed"])==s)) for s in [42,123,2025]]
    ax.plot([42,123,2025],vals,marker="o",label=display[m])
ax.set_xticks([42,123,2025],["42","123","2025"])
ax.set_xlabel("Random seed");ax.set_ylabel("Held-out Macro IoU")
ax.set_title("Inter-seed variation; proposed model is not best at every seed")
ax.legend(ncol=2,loc="best");ax.grid(alpha=.25);fig.tight_layout();save(fig,RESULT/"Figure_7_Seed_Comparison_Macro_IoU")

# Figure 9 – class-specific IoU heat map (mean across 3 seeds).
fig,ax=plt.subplots(figsize=(9.6,5.2))
mat=np.array([[statistics.mean(float(r[f"C{c}_IoU"]) for r in seeds if r["model"]==m) for c in range(5)] for m in model_order])
img=ax.imshow(mat,aspect="auto",cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(5),[f"C{i}" for i in range(5)])
ax.set_yticks(range(6),[display[m] for m in model_order])
for i in range(6):
    for j in range(5):
        ax.text(j,i,f"{mat[i,j]:.3f}",ha="center",va="center",fontsize=9,color="white" if mat[i,j]>.52 else cols["dark"])
ax.set_title("Per-class held-out IoU (mean of 3 seeds); rare C4 almost undetected")
fig.colorbar(img,ax=ax,label="IoU")
fig.tight_layout();save(fig,RESULT/"Figure_9_Per_Class_IoU")
print("FIGURES_DONE","2 updated schematics, 4 quantitative graphics, png+svg",flush=True)
