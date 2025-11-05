# Image Mask Crop - 图像遮罩裁剪

**节点功能：** `Image Mask Crop` 基于遮罩的边界框对图像与遮罩进行裁剪，或保持原尺寸；可选输出 RGBA（alpha 由遮罩提供）。支持批次循环与尺寸对齐。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 输入图像批次 |
| `mask` | 必选 | MASK | - | - | 输入遮罩批次 |
| `output_crop` | 必选 | BOOLEAN | True | True/False | True 时按遮罩 bbox 裁剪；False 时保持原尺寸 |
| `output_alpha` | 必选 | BOOLEAN | False | True/False | True 时输出 RGBA（alpha = mask），否则输出 RGB |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 裁剪或原尺寸的图像；`output_alpha=True` 时输出 RGBA |
| `mask` | MASK | 裁剪或尺寸对齐后的遮罩（0–1） |
