"""Create representative calibrated experimental dataset figure from actual Himawari/IMERG NPZ.
No synthetic pixels, no model inference. Single real 2024-05-03 source pair shown.
"""
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
R=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
F=R/"processed/aligned_multievent_2024/20240503_0210_B13_BT_IMERG_V07B.npz"
OUT=R/"github_release/OCSF-RainNet/figures/methods/Figure_1_Calibrated_2024_Representative_Scene"
d=np.load(F,allow_pickle=False)
bt=d["brightness_temperature_K"];rate=d["rain_rate_mm_hr"];labels=d["y"]
lat=d["lat"];lon=d["lon"]
valid=labels!=255
count=np.bincount(labels[valid].astype(int),minlength=5)
assert bt.shape==rate.shape==labels.shape==(185,460)
assert np.isfinite(bt).mean()>.99
fig=plt.figure(figsize=(16,9))
gs=fig.add_gridspec(2,2,width_ratios=[1,1.05],height_ratios=[1,1])
ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);ax3=fig.add_subplot(gs[1,0]);ax4=fig.add_subplot(gs[1,1])
extent=[float(lon.min())-.05,float(lon.max())+.05,float(lat.min())-.05,float(lat.max())+.05]
a=ax1.imshow(bt,cmap="gray_r",extent=extent,aspect="auto",vmin=180,vmax=320)
ax1.set_title("(a) Calibrated Himawari-9 AHI B13 brightness temperature")
fig.colorbar(a,ax=ax1,label="Brightness temperature (K)",shrink=.8)
b=ax2.imshow(rate,cmap="viridis",extent=extent,aspect="auto",vmin=0,vmax=20)
ax2.set_title("(b) Matched NASA IMERG Final V07B precipitation estimate")
fig.colorbar(b,ax=ax2,label="Estimated precipitation rate (mm h$^{-1}$)",shrink=.8,extend="max")
cmap=ListedColormap(["#abb1b9","#3b82f6","#1f9e89","#ffb030","#d83b32"])
c=ax3.imshow(np.ma.masked_where(~valid,labels),cmap=cmap,norm=BoundaryNorm(np.arange(-.5,5.5,1),5),
             extent=extent,aspect="auto")
ax3.set_title("(c) Deterministic five-class target (C0–C4)")
cb=fig.colorbar(c,ax=ax3,ticks=np.arange(5),label="Intensity category",shrink=.8)
cb.ax.set_yticklabels([f"C{i}" for i in range(5)])
bars=ax4.bar([f"C{i}" for i in range(5)],count,color=cmap.colors)
ax4.set_title("(d) Single-scene class prevalence (grid cells)")
ax4.set_ylabel("Number of valid IMERG cells")
for bar,n in zip(bars,count):
    ax4.annotate(f"{n:,} ({n/count.sum()*100:.2f}%)",(bar.get_x()+bar.get_width()/2,n),
                 xytext=(0,3),textcoords="offset points",ha="center",fontsize=8)
for ax in (ax1,ax2,ax3):
    ax.set_xlabel("Longitude (°E)");ax.set_ylabel("Latitude (°N)")
fig.suptitle("Representative source–target observation: 3 May 2024, AHI 02:10 UTC / IMERG 02:00–02:30 UTC",
            fontsize=13,fontweight="bold")
fig.text(.5,.01,"Satellite-estimated label field, not independent rain-gauge truth and not a model prediction.",
         ha="center",fontsize=9)
fig.tight_layout(rect=[0,.035,1,.95])
fig.savefig(str(OUT)+".png",dpi=320,bbox_inches="tight")
fig.savefig(str(OUT)+".svg",bbox_inches="tight")
plt.close(fig)
print("CALIBRATED_FIGURE_1",str(OUT),count.tolist(),flush=True)
