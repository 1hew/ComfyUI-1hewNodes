# Image BBox Mask Crop - 图像边界框遮罩裁剪

**节点功能：** `Image BBox Mask Crop`节点根据边界框遮罩信息批量裁剪图像，支持两种输出模式：完整区域或仅白色区域（带alpha通道）。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要裁剪的图像 |
| `mask` | 必选 | MASK | - | - | 定义边界框的遮罩 |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | 输出模式：bbox_rgb（完整区域）或mask_rgba（带alpha的遮罩区域） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `cropped_image` | IMAGE | 裁剪后的图像 |

## 功能说明

### 输出模式
- **BBox RGB模式**：输出完整的边界框区域作为RGB图像
- **Mask RGBA模式**：仅输出遮罩区域，带alpha通道实现透明效果