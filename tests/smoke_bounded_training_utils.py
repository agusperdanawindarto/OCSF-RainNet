"""CPU-only engineering smoke tests for resource-bounded data and resume callbacks.

IMPORTANT: zero scientific model training/results are created. Synthetic tensors
exist only in RAM and temporary files under project logs and are deleted.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import tensorflow as tf
from resource_limits import configure_resource_limits, require_paper_gate
from bounded_training_utils import make_dataset, WallClockLimiter, safe_callbacks

ROOT=Path(r"D:\DISERTASI_S3\SEGMENTASI_HUJAN")

def main():
    budget=configure_resource_limits(tf,mode="screening")
    assert budget["cpu_threads_max"]==4
    inputs=sorted((ROOT/"processed"/"aligned").glob("*_B13_IMERG_V07.npz"))
    assert len(inputs)==7
    sample_train=next(iter(make_dataset(inputs[:1],batch_size=2,shuffle=False,mode="train")))
    sample_eval=next(iter(make_dataset(inputs[:1],batch_size=1,mode="eval")))
    x,y=sample_train;xe,ye=sample_eval
    assert tuple(x.shape)==(2,128,128,3)
    assert tuple(y.shape)==(2,128,128)
    assert tuple(xe.shape)==(1,192,464,3)
    assert tuple(ye.shape)==(1,192,464)
    assert (ye.numpy()==255).any() and np.isfinite(x.numpy()).all()
    print("REAL_DATA_PATCH_LOADER_PASS","train",x.shape,"evaluation",xe.shape,flush=True)
    try:
        require_paper_gate()
    except RuntimeError as e:
        print("PAPER_TRAINING_BLOCKED_AS_REQUIRED",type(e).__name__,flush=True)
    else:
        raise AssertionError("Training should not be authorized with only 2 days and CPU-only TF")

    # A tiny in-memory toy convolution model tests callback wiring/reload,
    # NOT the proposed OCSF-RainNet model or rainfall performance.
    with TemporaryDirectory(prefix="toy_recovery_",dir=str(ROOT/"logs")) as temp:
        tf.keras.utils.set_random_seed(42)
        model=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(16,16,3)),
            tf.keras.layers.Conv2D(5,1,activation="softmax")
        ])
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")
        toy_x=np.zeros((2,16,16,3),dtype="float32")
        toy_y=np.zeros((2,16,16),dtype="int32")
        limiter=WallClockLimiter(max_minutes=1)
        model.fit(toy_x,toy_y,validation_data=(toy_x,toy_y),epochs=1,batch_size=1,
                  callbacks=safe_callbacks(Path(temp)/"toy_run",limiter,patience=4),
                  verbose=0)
        path=Path(temp)/"toy_run"/"best.weights.h5"
        assert path.is_file() and not limiter.hit_limit
        old=model(toy_x,training=False).numpy()
        model.set_weights([np.zeros_like(w) for w in model.get_weights()])
        model.load_weights(str(path))
        new=model(toy_x,training=False).numpy()
        np.testing.assert_allclose(old,new,atol=1e-6)
        print("TOY_CHECKPOINT_RELOAD_PASS (no rainfall training)",flush=True)
    print("SMOKE_TESTS_COMPLETED_TEMP_ARTIFACTS_DELETED",flush=True)

if __name__=="__main__":
    main()
