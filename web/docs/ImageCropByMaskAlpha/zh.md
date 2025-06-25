# Image Crop by Mask Alpha

**节点功能：** `Image Crop by Mask Alpha`节点用于根据边界框遮罩信息批量裁剪图像，支持两种输出模式：完整区域或仅白色区域（带alpha通道），常用于智能图像裁剪和区域提取。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要裁剪的图像（支持RGB和RGBA格式） |
| `mask` | 必选 | MASK | - | - | 用于确定裁剪边界的遮罩 |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | 输出模式：bbox_rgb（RGB格式的完整裁剪区域），mask_rgba（RGBA格式的仅白色区域带alpha通道） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `cropped_image` | IMAGE | 裁剪后的图像结果（bbox_rgb模式为RGB格式，mask_rgba模式为RGBA格式） |
| `cropped_mask` | MASK | 对应边界框区域的裁剪遮罩 |

## 功能特性

- **智能通道处理**：在bbox_rgb模式下自动将4通道RGBA输入转换为3通道RGB输出
- **双输出模式**：
  - `bbox_rgb`：输出RGB格式的完整裁剪区域（3通道）
  - `mask_rgba`：输出带alpha通道的遮罩区域（4通道）
- **批量处理**：支持同时处理多个图像和遮罩
- **自动填充**：通过填充处理不同尺寸的图像以统一尺寸
- **遮罩输出**：两种输出模式都提供裁剪后的遮罩区域