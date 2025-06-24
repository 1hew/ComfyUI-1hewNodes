# Image Crop by Mask Alpha

**节点功能：** `Image Crop by Mask Alpha`节点用于根据边界框遮罩信息批量裁剪图像，支持两种输出模式：完整区域或仅白色区域（带alpha通道），常用于智能图像裁剪和区域提取。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要裁剪的图像 |
| `mask` | 必选 | MASK | - | - | 用于确定裁剪边界的遮罩 |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | 输出模式：bbox_rgb（完整裁剪区域），mask_rgba（仅白色区域带alpha通道） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `cropped_image` | IMAGE | 裁剪后的图像结果 |