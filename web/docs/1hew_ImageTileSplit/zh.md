# Image Tile Split - 网格切片与重叠

**节点功能：** `Image Tile Split` 将单张图像切分为带可选重叠的网格切片。支持自动网格估算、命名预设`、`actual_crop_size`；可直接用于 `Image Tile Merge` 合并。
- 输入为批次时仅使用第一张；建议输入单张批次以保持一致。