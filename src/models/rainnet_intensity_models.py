"""RainNet-Intensity: faithful original RainNet backbone + proposed extensions.

Evidence:
- Windarto, Yuhandri & Bukhori (2024), ICIC Express Letters 18(4), 313-323,
  DOI: 10.24507/icicel.18.04.313.
- Original Keras model: 4 residual encoders [4,8,16,32], bridge 64,
  4 residual decoders, sigmoid 1-channel binary head, 142617 parameters.
- This implementation LOADS THE REAL 2024 checkpoint and REUSES its layers,
  with new multiclass head(s). It never overwrites dissertation checkpoint.
- Proposed OCSF-RainNet = Ordinal Cross-Scale Fusion RainNet:
    (A) Context-gated cross-scale residual fusion of skip features; and
    (B) ordered cumulative exceedance head for 5 rainfall-intensity classes.
- Baselines: U-Net softmax, residual attention U-Net style, original RainNet
  encoder/decoder with softmax head, + separate ablations.
- No empirical superiority or rainfall forecast is claimed until evaluated
  against verified IMERG precipitation labels and ideally independent BMKG.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import tensorflow as tf

BASE = Path(r"D:\DISERTASI_S3")
ROOT = BASE / "SEGMENTASI_HUJAN"
V1 = Path(os.getenv("RAINNET_V1_ROOT", str(BASE/"PROGRAM_RAINNET")))
LOCAL_PRETRAINED = ROOT / "models" / "pretrained_rainnet_2024"
# A byte-for-byte, SHA-256-verified copy of 2024 checkpoint + Python class
# makes this project independently reproducible without modifying RainNet V1.
CHECKPOINT = (LOCAL_PRETRAINED / "rainnet_binary_best_05.h5"
              if (LOCAL_PRETRAINED / "rainnet_binary_best_05.h5").is_file()
              else V1 / "models" / "selected" / "rainnet_binary_best_05.h5")
ARCH = (LOCAL_PRETRAINED
        if (LOCAL_PRETRAINED / "rainNetModel2labels.py").is_file()
        else V1 / "app" / "architectures")
ORIGINAL_CHECKPOINT_SHA256 = "71e0e6d7e0fa9fa6f541c89f1bfcc57d701bd94730e0508c2dbf45c341de575f"


def _archive_module():
    if not ARCH.is_dir():
        raise FileNotFoundError("Original RainNet architectures directory missing: "+str(ARCH))
    if str(ARCH) not in sys.path:
        sys.path.insert(0,str(ARCH))
    import rainNetModel2labels as original
    return original


def load_2024_checkpoint():
    """Strict provenance check; original stored HDF5 must exist unchanged."""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("2024 checkpoint not found: "+str(CHECKPOINT))
    actual_hash=pretrained_sha256()
    if actual_hash!=ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError("2024 RainNet checkpoint integrity failure: "+actual_hash)
    original = _archive_module()
    model = tf.keras.models.load_model(
        str(CHECKPOINT),
        custom_objects={name:getattr(original,name)
                        for name in ("Encoder","Bridge","Decoder","dice_coeff")},
        compile=False)
    if model.input_shape != (None,400,400,3):
        raise ValueError("Not the original 400x400x3 RainNet input")
    if model.output_shape != (None,400,400,1):
        raise ValueError("Original model should have a binary output head")
    if model.count_params() != 142617:
        raise ValueError("Checkpoint parameter count differs from verified dissertation model")
    # Keras auto-renames custom blocks after repeated loads in one process.
    # The robust provenance check relies on types/order/filters, NOT names.
    expected_types=["Encoder"]*4+["Bridge"]+["Decoder"]*4+["Conv2D"]
    real_layers=[l for l in model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
    actual_types=[type(l).__name__ for l in real_layers]
    actual_filters=[int(l.filter_num) for l in real_layers[:-1]]
    expected_filters=[4,8,16,32,64,32,16,8,4]
    if actual_types!=expected_types or actual_filters!=expected_filters:
        raise ValueError("Checkpoint structure changed: "+
                         repr(list(zip(actual_types,actual_filters))))
    return model


def pretrained_sha256():
    checksum=hashlib.sha256()
    with CHECKPOINT.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            checksum.update(block)
    return checksum.hexdigest()


@tf.keras.utils.register_keras_serializable(package="RainNetIntensity")
class OrderedFiveClassDistribution(tf.keras.layers.Layer):
    """Five-class output with constrained monotonic cumulative probabilities.

    A pixel has exceedance probabilities q_k=P(Y >= k+1), k=0..3:
    z0=raw[...,0], z1=z0-softplus(raw[...,1]), etc.
    By construction q0 >= q1 >= q2 >= q3, and:
    p0=1-q0, p1=q0-q1, p2=q1-q2, p3=q2-q3, p4=q3.
    The ordinal structure does NOT imply that rain intensity increases
    spatially; it just enforces consistent per-pixel severity thresholds.
    """
    def call(self,raw):
        init=raw[...,:1]
        gaps=tf.nn.softplus(raw[...,1:])  # non-negative
        logits=tf.concat([init,init-tf.cumsum(gaps,axis=-1)],axis=-1)
        exceedance=tf.math.sigmoid(logits)
        return tf.concat([
            1.0-exceedance[...,:1],
            exceedance[...,:-1]-exceedance[...,1:],
            exceedance[...,-1:]],axis=-1)

    def get_config(self):
        return super().get_config()


def _cross_scale_fusion(last_decoded,skips):
    """Proposed fusion: shallow texture + broader context + decoded features."""
    # Original skip resolutions: 1x, 1/2x, 1/4x input.
    f0=tf.keras.layers.Conv2D(8,1,padding="same",name="csf_high_detail")(skips[0])
    f1=tf.keras.layers.Conv2D(8,1,padding="same",name="csf_mid_context")(skips[1])
    f1=tf.keras.layers.UpSampling2D(2,interpolation="bilinear",name="csf_mid_up")(f1)
    f2=tf.keras.layers.Conv2D(8,1,padding="same",name="csf_low_context")(skips[2])
    f2=tf.keras.layers.UpSampling2D(4,interpolation="bilinear",name="csf_low_up")(f2)
    decoded=tf.keras.layers.Conv2D(8,3,padding="same",name="csf_decoded")(last_decoded)
    merged=tf.keras.layers.Concatenate(name="csf_multiscale_concat")([f0,f1,f2,decoded])
    # Spatial gate conditioned jointly on high/middle/low resolution evidence.
    att=tf.keras.layers.Conv2D(1,3,activation="sigmoid",padding="same",
                               name="csf_spatial_gate")(merged)
    features=tf.keras.layers.Conv2D(16,3,padding="same",activation="relu",
                                     name="csf_fused")(merged)
    gated=tf.keras.layers.Multiply(name="csf_gated")([features,att])
    skip=tf.keras.layers.Conv2D(16,1,padding="same",
                                name="csf_decoder_projection")(last_decoded)
    return tf.keras.layers.Add(name="csf_residual_fusion")([gated,skip])


def build_rainnet_variant(variant="proposed",pretrained=True,
                          freeze_backbone=True):
    """Four variants preserve *exact same* original 2024 residual backbone.

    variant:
      plain     = 2024 RainNet (same core), 5-class softmax head.
      fusion    = plain + cross-scale gated fusion.
      ordinal   = plain + ordinal cumulative 5-class head.
      proposed  = fusion + ordinal head (OCSF-RainNet).

    Dynamic input H,W must be multiples of 16 to match 4 pool/upsampling levels.
    The historic checkpoint's fixed 400x400 input belongs to the original graph,
    but the reused convolutional layers accept variable dimensions.
    """
    if variant not in ("plain","fusion","ordinal","proposed"):
        raise ValueError("Unknown RainNet variant: "+str(variant))
    if not pretrained:
        raise ValueError("Only checkpoint-grounded RainNet is supported in this function")
    source=load_2024_checkpoint()
    inp=tf.keras.Input(shape=(None,None,3),name="Himawari_B13_on_IMERG_grid")
    pooled=inp
    skips=[]
    blocks=[l for l in source.layers if type(l).__name__=="Encoder"]
    bridge=next(l for l in source.layers if type(l).__name__=="Bridge")
    decoders=[l for l in source.layers if type(l).__name__=="Decoder"]
    for enc in blocks:
        pooled,sk=enc(pooled)
        skips.append(sk)
    pooled=bridge(pooled)
    for dec,skip in zip(decoders,reversed(skips)):
        pooled=dec(pooled,skip)
    backbone_layers=blocks+[bridge]+decoders
    for layer in backbone_layers:
        layer.trainable=not freeze_backbone
    feature=(_cross_scale_fusion(pooled,skips) if variant in ("fusion","proposed")
             else pooled)
    if variant in ("ordinal","proposed"):
        raw=tf.keras.layers.Conv2D(4,1,padding="same",
                        name="ordinal_exceedance_parameters")(feature)
        output=OrderedFiveClassDistribution(name="ordered_5class_probabilities")(raw)
    else:
        output=tf.keras.layers.Conv2D(5,1,activation="softmax",
                                     name="five_class_softmax")(feature)
    model=tf.keras.Model(inp,output,name="RainNet_"+variant+"_5class")
    model.rainnet_checkpoint_sha256=pretrained_sha256()
    model.rainnet_backbone_layers=backbone_layers
    return model


def conv_block(x,n,residual=False,prefix="c"):
    y=tf.keras.layers.Conv2D(n,3,padding="same",activation="relu",name=prefix+"_c1")(x)
    y=tf.keras.layers.Conv2D(n,3,padding="same",name=prefix+"_c2")(y)
    if residual:
        x=tf.keras.layers.Conv2D(n,1,padding="same",name=prefix+"_proj")(x)
        y=tf.keras.layers.Add(name=prefix+"_res")([x,y])
    return tf.keras.layers.Activation("relu",name=prefix+"_relu")(y)


def build_unet_baseline(residual_attention=False):
    """5-class U-Net baseline, or RA-U-Net-style comparator.

    RA variant is a transparent *style* reimplementation, not authors'
    code or claimed exact replica of Ji et al. (Remote Sensing 2024).
    Both are same B13 input & label target as proposed RainNet.
    """
    inp=tf.keras.Input((None,None,3),name="B13_input")
    widths=[8,16,32,64]  # lightweight comparison
    h=inp
    skips=[]
    for i,n in enumerate(widths):
        h=conv_block(h,n,residual_attention,prefix=f"u_enc{i}")
        skips.append(h)
        h=tf.keras.layers.MaxPool2D(name=f"u_pool{i}")(h)
    h=conv_block(h,128,residual_attention,prefix="u_bridge")
    for i,(n,sk) in enumerate(zip(reversed(widths),reversed(skips))):
        h=tf.keras.layers.Conv2DTranspose(n,2,strides=2,padding="same",
                                          name=f"u_up{i}")(h)
        if residual_attention:
            gate=tf.keras.layers.Concatenate(name=f"u_att_concat{i}")([h,sk])
            gate=tf.keras.layers.Conv2D(1,1,activation="sigmoid",
                                       name=f"u_att_gate{i}")(gate)
            sk=tf.keras.layers.Multiply(name=f"u_attended_skip{i}")([sk,gate])
        h=tf.keras.layers.Concatenate(name=f"u_skip{i}")([h,sk])
        h=conv_block(h,n,residual_attention,prefix=f"u_dec{i}")
    outputs=tf.keras.layers.Conv2D(5,1,activation="softmax",
                                   name="u_multiclass_softmax")(h)
    return tf.keras.Model(inp,outputs,name="RA_UNet_style" if residual_attention else "UNet")


def masked_weighted_ordinal_loss(class_weights=None,ordinal_weight=0.12):
    """Masked class-balanced loss + mean ordinal severity-distance penalty.

    Numeric class weights MUST be estimated from train split only, not test.
    Applies to softmax *and* proposed ordered probabilities for fair training.
    """
    if class_weights is None:
        class_weights=[1.0]*5
    weights=tf.constant([float(w) for w in class_weights],tf.float32)
    if len(class_weights)!=5 or min(class_weights)<=0:
        raise ValueError("Exactly five positive train-only class weights expected")

    def loss(y_true,probabilities):
        target=tf.cast(y_true,tf.int32)
        valid=tf.not_equal(target,255)
        labels=tf.where(valid,target,0)
        prob=tf.clip_by_value(probabilities,1e-7,1.0)
        ce=tf.keras.losses.sparse_categorical_crossentropy(labels,prob)
        w=tf.gather(weights,labels)*tf.cast(valid,tf.float32)
        ce_loss=tf.reduce_sum(ce*w)/tf.maximum(tf.reduce_sum(w),1.)
        expected=tf.reduce_sum(prob*tf.cast(tf.range(5),tf.float32),axis=-1)
        ordinal=tf.abs(expected-tf.cast(labels,tf.float32))/4.
        ord_loss=tf.reduce_sum(ordinal*tf.cast(valid,tf.float32))/tf.maximum(
            1.,tf.reduce_sum(tf.cast(valid,tf.float32)))
        return ce_loss+ordinal_weight*ord_loss

    return loss


def empirical_train_class_weights(counts):
    """Square-root inverse frequencies, capped to stabilize very rare classes."""
    import numpy as np
    counts=np.asarray(counts,dtype=np.float64)
    if counts.shape!=(5,) or not (counts>0).all():
        raise ValueError("All five classes must have examples in TRAIN split")
    freq=counts/counts.sum()
    raw=np.sqrt(1./freq)
    raw=np.clip(raw,1.0,12.0)
    return (raw/np.mean(raw)).tolist()
