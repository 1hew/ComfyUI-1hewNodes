# Image Paste by BBox Mask - 图像边界框遮罩粘贴

**节点功能：** `Image Paste by BBox Mask`节点用于将处理后的裁剪图像根据边界框遮罩信息粘贴回原始图像的位置，支持多种混合模式和不透明度控制。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `base_image` | 必选 | IMAGE | - | - | 要粘贴到的基础图像 |
| `cropped_image` | 必选 | IMAGE | - | - | 要粘贴的裁剪图像 |
| `bbox_mask` | 必选 | MASK | - | - | 指示粘贴位置的边界框遮罩 |
| `blend_mode` | - | COMBO[STRING] | normal | normal, multiply, screen, overlay, soft_light, difference | 粘贴时的混合模式 |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | 粘贴时的不透明度 |
| `cropped_mask` | 可选 | MASK | - | - | 裁剪图像的可选遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 粘贴裁剪内容后的最终图像 |