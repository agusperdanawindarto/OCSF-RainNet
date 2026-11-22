"""Bounded patch loader and wall-clock/callback helpers for RainNet experiments.

No training is launched by importing this file. CPU/GPU setup and scientific
readiness must be checked separately before calling model.fit. Values are
specified in resource_experiment_preflight.json.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import tensorflow as tf

ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")
PATCH_SIZE=128
MAX_PATCHES_PER_SCENE=6

def make_dataset(paths,batch_size=2,shuffle=False,seed=42,mode="eval"):
    """Read one NPZ at a time. Oversample rare classes in *training only*.

    - mode train -> deterministic 128x128 event-aware patches (6 / scene).
    - mode eval -> complete original aligned scene padded to 16, no sampling.
    """
    if mode not in ("train","eval"):
        raise ValueError("mode must be train or eval")
    paths=[Path(p) for p in paths]
    if not paths:
        raise ValueError("Empty aligned data list")
    with np.load(paths[0],allow_pickle=False) as d:
        H,W=d["y"].shape
    Hp=H+(-H)%16;Wp=W+(-W)%16
    if mode=="train" and (H<PATCH_SIZE or W<PATCH_SIZE):
        raise ValueError("IMERG aligned scene smaller than 128px patch")
    rng=np.random.default_rng(seed)
    def iterate():
        file_order=list(paths)
        if shuffle:
            rng.shuffle(file_order)
        for path in file_order:
            with np.load(path,allow_pickle=False) as d:
                x=np.asarray(d["X"],np.float32)
                y=np.asarray(d["y"],np.int32)
            if x.shape!=(H,W,3) or y.shape!=(H,W):
                raise ValueError("Inconsistent spatial shapes in aligned dataset")
            if not np.isfinite(x).all() or x.min()<-0.001 or x.max()>1.001:
                raise ValueError("Himawari images should be normalized to [0,1]")
            if mode=="eval":
                yield (np.pad(x,((0,Hp-H),(0,Wp-W),(0,0)),constant_values=0),
                       np.pad(y,((0,Hp-H),(0,Wp-W)),constant_values=255))
                continue
            # Pick centres from moderate/heavy IMERG labels for half the patches.
            # Intensity-class sampling does NOT change validation/test prevalence.
            candidate=np.argwhere((y>=2)&(y<=4))
            for i in range(MAX_PATCHES_PER_SCENE):
                if i%2==0 and len(candidate):
                    cy,cx=candidate[int(rng.integers(len(candidate)))]
                    top=int(np.clip(cy-PATCH_SIZE//2,0,H-PATCH_SIZE))
                    left=int(np.clip(cx-PATCH_SIZE//2,0,W-PATCH_SIZE))
                else:
                    top=int(rng.integers(0,H-PATCH_SIZE+1))
                    left=int(rng.integers(0,W-PATCH_SIZE+1))
                xx=x[top:top+PATCH_SIZE,left:left+PATCH_SIZE,:]
                yy=y[top:top+PATCH_SIZE,left:left+PATCH_SIZE]
                assert xx.shape==(PATCH_SIZE,PATCH_SIZE,3)
                assert yy.shape==(PATCH_SIZE,PATCH_SIZE)
                yield xx,yy
    sh=(PATCH_SIZE,PATCH_SIZE) if mode=="train" else (Hp,Wp)
    signature=(tf.TensorSpec(sh+(3,),tf.float32),tf.TensorSpec(sh,tf.int32))
    return tf.data.Dataset.from_generator(iterate,output_signature=signature).batch(
        batch_size,drop_remainder=False).prefetch(1)


class WallClockLimiter(tf.keras.callbacks.Callback):
    """Stop incomplete runs rather than silently treating timeout as convergence."""
    def __init__(self,max_minutes=90,start_time=None):
        super().__init__()
        self.time_budget_s=float(max_minutes)*60.0
        self.started=time.monotonic() if start_time is None else start_time
        self.hit_limit=False

    def on_train_batch_end(self,batch,logs=None):
        if time.monotonic()-self.started>=self.time_budget_s:
            self.hit_limit=True
            self.model.stop_training=True
            print("RESOURCE_WALL_CLOCK_EXCEEDED: mark run INCOMPLETE; no paper score.",flush=True)


def safe_callbacks(run_dir,limiter,patience=4):
    """Resume and best-checkpoint hooks for future validated training runs."""
    run_dir=Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    return [
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(run_dir/"backup")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir/"best.weights.h5"),monitor="val_loss",
            save_best_only=True,save_weights_only=True,verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=patience,restore_best_weights=True),
        limiter
    ]
