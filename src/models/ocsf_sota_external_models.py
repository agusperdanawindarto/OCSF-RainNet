"""Auditable independent architecture comparators for OCSF-RainNet.

Both models use uninitialized weights and the *identical* OCSF data, loss, training
and held-out metric code. SegFormer-B0: task-matched MixTransformer-B0 topology;
DeepLabv3+: task-matched depthwise CNN/ASPP encoder-decoder topology. Neither is
claimed to reproduce an external paper's trained checkpoint or published scores.
No GPU operations on Windows; code is executed only on Kaggle.
"""
import tensorflow as tf
L=tf.keras.layers

def _conv(x,c,k=3,s=1,r=1,name=None):
    x=L.Conv2D(c,k,strides=s,dilation_rate=r,padding="same",use_bias=False,name=name)(x)
    x=L.BatchNormalization()(x)
    return L.Activation("relu")(x)

def _sep(x,c,k=3,s=1,r=1):
    x=L.SeparableConv2D(c,k,strides=s,dilation_rate=r,padding="same",use_bias=False)(x)
    x=L.BatchNormalization()(x)
    return L.Activation("relu")(x)

def _resize_to(src,ref):
    return L.Lambda(lambda z: tf.image.resize(z[0],tf.shape(z[1])[1:3],method="bilinear"))([src,ref])

def build_deeplabv3plus():
    """Task-matched DeepLabv3+-style ASPP/depthwise CNN, random initialization."""
    inp=L.Input(shape=(None,None,3),name="calibrated_ahi_b13")
    x=_conv(inp,32,3,2)
    x=_sep(x,48)
    low=_sep(x,64,3,2)             # 1/4 low-level features
    x=_sep(low,96,3,2)             # 1/8
    x=_sep(x,160,3,2)             # 1/16
    x=_sep(x,192)
    branches=[_conv(x,64,1,1)]
    for rate in (6,12,18):
        branches.append(_sep(x,64,3,1,rate))
    context=L.GlobalAveragePooling2D(keepdims=True)(x)
    context=_conv(context,64,1,1)
    context=_resize_to(context,x)
    branches.append(context)
    x=L.Concatenate(axis=-1)(branches)
    x=_conv(x,128,1,1)
    x=_resize_to(x,low)
    skip=_conv(low,32,1,1)
    x=L.Concatenate(axis=-1)([x,skip])
    x=_sep(x,96)
    x=_sep(x,96)
    logits=L.Conv2D(5,1,padding="same")(x)
    logits=L.UpSampling2D(size=(4,4),interpolation="bilinear")(logits)
    out=L.Softmax(axis=-1,name="five_class_probabilities")(logits)
    return tf.keras.Model(inp,out,name="task_matched_DeepLabv3plus")

class MiTB0Block(L.Layer):
    """MiT stage: reduced-resolution MHSA and convolutional MixFFN."""
    def __init__(self,dim,heads,sr_ratio,mlp_ratio=4,**kwargs):
        super().__init__(**kwargs)
        self.dim=dim
        self.sr_ratio=sr_ratio
        self.norm1=L.LayerNormalization(epsilon=1e-6)
        self.normkv=L.LayerNormalization(epsilon=1e-6)
        self.sr=L.Conv2D(dim,kernel_size=sr_ratio,strides=sr_ratio,padding="same") if sr_ratio>1 else None
        self.att=L.MultiHeadAttention(num_heads=heads,key_dim=dim//heads,dropout=0.0)
        self.norm2=L.LayerNormalization(epsilon=1e-6)
        self.expand=L.Conv2D(dim*mlp_ratio,kernel_size=1,padding="same")
        self.dwc=L.DepthwiseConv2D(kernel_size=3,padding="same")
        self.project=L.Conv2D(dim,kernel_size=1,padding="same")
    def call(self,x,training=None):
        y=self.norm1(x)
        shp=tf.shape(y)
        q=tf.reshape(y,[shp[0],shp[1]*shp[2],self.dim])
        kv=self.normkv(self.sr(y)) if self.sr is not None else y
        kshp=tf.shape(kv)
        kv=tf.reshape(kv,[kshp[0],kshp[1]*kshp[2],self.dim])
        z=self.att(q,kv,training=training)
        z=tf.reshape(z,[shp[0],shp[1],shp[2],self.dim])
        x=x+z
        y=self.norm2(x)
        y=self.expand(y)
        y=self.dwc(y)
        y=tf.nn.gelu(y)
        y=self.project(y)
        return x+y

def build_segformer_b0():
    """SegFormer MiT-B0 [32,64,160,256], [2,2,2,2] + lightweight MLP decoder.
    Random initialization. Distinct from use of publicly pretrained checkpoints.
    """
    inp=L.Input(shape=(None,None,3),name="calibrated_ahi_b13")
    x=inp
    feats=[]
    cfg=[(32,1,8,4,7),(64,2,4,2,3),(160,5,2,2,3),(256,8,1,2,3)]
    for si,(dim,heads,sr,stride,kernel) in enumerate(cfg):
        x=L.Conv2D(dim,kernel_size=kernel,strides=stride,padding="same",name=f"mit_patch_s{si+1}")(x)
        x=L.LayerNormalization(epsilon=1e-6)(x)
        for bi in range(2):
            x=MiTB0Block(dim,heads,sr,name=f"mit_s{si+1}_block{bi+1}")(x)
        feats.append(x)
    high=feats[0]
    multiscale=[]
    for feat in feats:
        p=L.Conv2D(128,1,padding="same")(feat)
        p=_resize_to(p,high)
        multiscale.append(p)
    x=L.Concatenate(axis=-1)(multiscale)
    x=_conv(x,128,1,1)
    x=L.Conv2D(5,1,padding="same")(x)
    x=L.UpSampling2D(size=(4,4),interpolation="bilinear")(x)
    out=L.Softmax(axis=-1,name="five_class_probabilities")(x)
    return tf.keras.Model(inp,out,name="task_matched_SegFormer_B0")
