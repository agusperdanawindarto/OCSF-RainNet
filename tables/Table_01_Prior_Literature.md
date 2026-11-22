# Table 1. Related work / positioning (as surveyed in the original manuscript)

| Study | Source and target | Contribution | Difference from this work |
|---|---|---|---|
| Lagerquist et al. (2021) | Multitemporal Himawari-8; radar-defined convection | U-Net future convection coverage | Not current IMERG-derived intensity classes |
| Ji et al. (2024) | Six times × seven Himawari channels; future estimated rainfall | RA-UNet and future-rain regression | Different inputs, target and lead time |
| Li et al. (2024) | FY-4A; cloud masks | Dugs-UNet gated shape and fusion | Cloud regions, not ordinal rainfall rate |
| Windarto et al. (2024) | Himawari-derived binary cloud mask | Original pretrained residual RainNet | Direct architectural ancestor; five-class head is the control |
| Vélez-Hernández et al. (2025) | GOES-16; IMERG Early Run | Continuous rainfall estimation | Different sensor/product and regression target |
| Rouzegari et al. (2026) | IR+passive microwave; MRMS | Sensor fusion for rainfall | Independent radar data not available in this work |
| Liu et al. (2026) | Himawari visible and IR observations | Multitemporal QPE | Richer spectral and time information |
| Hong and Qiu (2026) | High-resolution remote-sensing land cover | Cross-scale gating and graph attention | Prior art for multiscale feature learning |
| Cao et al. (2020) | Age estimation | Rank-consistent ordinal regression | Prior ordinal theory, not satellite precipitation |
