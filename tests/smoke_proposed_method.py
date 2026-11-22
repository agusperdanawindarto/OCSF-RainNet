"""No-data/no-training smoke test on real pretrained RainNet checkpoint."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from rainnet_intensity_models import (CHECKPOINT,load_2024_checkpoint,
    build_rainnet_variant,build_unet_baseline,masked_weighted_ordinal_loss,
    empirical_train_class_weights,pretrained_sha256)

tf.keras.utils.set_random_seed(42)
src=load_2024_checkpoint()
before=next(l for l in src.layers if type(l).__name__=="Encoder").get_weights()[0]
print("SOURCE",src.count_params(),"SHA256",pretrained_sha256(),flush=True)
dummy=tf.random.uniform((1,64,64,3),seed=42)
records=[]
for kind in ("plain","fusion","ordinal","proposed"):
    model=build_rainnet_variant(kind,pretrained=True,freeze_backbone=True)
    out=model(dummy,training=False).numpy()
    assert out.shape==(1,64,64,5)
    assert np.isfinite(out).all()
    assert (out>=-1e-6).all() and (out<=1.00001).all()
    np.testing.assert_allclose(out.sum(axis=-1),1.,atol=1e-5)
    np.testing.assert_array_equal(before,next(l for l in model.layers if type(l).__name__=="Encoder").get_weights()[0])
    if kind in ("ordinal","proposed"):
        # cumulative probability P(Y >= k) is monotone by construction
        q=np.stack([out[...,k:].sum(axis=-1) for k in range(1,5)],axis=-1)
        assert (np.diff(q,axis=-1)<1e-6).all()
    labels=tf.zeros((1,64,64),dtype=tf.int32)
    loss=masked_weighted_ordinal_loss()(labels,tf.convert_to_tensor(out))
    assert np.isfinite(float(loss.numpy()))
    counts={"variant":kind,"parameters":model.count_params(),"output":list(out.shape)}
    records.append(counts)
    print("CHECKPOINT_RAINNET_VARIANT_OK",counts,flush=True)
    del model,out
    tf.keras.backend.clear_session()
for residual in (False,True):
    model=build_unet_baseline(residual_attention=residual)
    out=model(dummy,training=False).numpy()
    np.testing.assert_allclose(out.sum(-1),1.,atol=1e-5)
    name="RA-UNet-style" if residual else "U-Net"
    counts={"variant":name,"parameters":model.count_params(),"output":list(out.shape)}
    records.append(counts)
    print("PUBLISHED_STYLE_BASELINE_OK",counts,flush=True)
    del model
    tf.keras.backend.clear_session()
weights=empirical_train_class_weights([1000,500,200,30,10])
assert len(weights)==5
assert np.isfinite(weights).all()
print("SMOKE_TEST_COMPLETED",len(records),"architectures; NO_LABELS, NO_TRAINING",flush=True)
report=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN\metadata\inventories")/"proposed_architecture_smoke.json"
report.write_text(json.dumps({"provenance":"RainNet 2024 pretrained HDF5",
                              "checkpoint_sha256":pretrained_sha256(),
                              "models":records,
                              "trained":False,"imerg_labels":False},
                             indent=2),encoding="utf-8")
