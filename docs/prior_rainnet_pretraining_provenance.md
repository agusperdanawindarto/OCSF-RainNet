# Original published RainNet 2024 checkpoint (preserved copy)

Research: Windarto, Agus Perdana; Yuhandri; Bukhori (2024).
*A Novel Convective Cloud Segmentation for Heavy Rainfall Potential Assessment Using Deep Learning*.
ICIC Express Letters 18(4), 313–323. DOI https://doi.org/10.24507/icicel.18.04.313

Original local files:
- D:\DISERTASI_S3\PROGRAM_RAINNET\models\selected\rainnet_binary_best_05.h5
- D:\DISERTASI_S3\PROGRAM_RAINNET\app\architectures\rainNetModel2labels.py

Files in this directory are byte-for-byte preserved copies (not newly trained
models), copied to SEGMENTASI_HUJAN to make the proposed experiment self-contained.

SHA-256 of `rainnet_binary_best_05.h5`:
`71e0e6d7e0fa9fa6f541c89f1bfcc57d701bd94730e0508c2dbf45c341de575f`

SHA-256 of `rainNetModel2labels.py`:
`0a2f213761819595528599e74b53b3019fb633134aa95829fa08d0da921f2121`

Original checkpoint: input 400x400x3, output 400x400x1, 142617 parameters.
Original class blocks Encoder, Bridge and Decoder are preserved. The
OCSF-RainNet proposed architecture reuses the 2024 residual blocks and weights,
adds cross-scale fusion and ordinal five-class output. No changes are made
to the original published checkpoint or source code.

These are research files, not public operational flood warnings.
