# Image BBox Paste - 图像边界框粘贴

**节点功能：** `Image BBox Paste`节点将处理后的裁剪图像粘贴回原始图像的指定位置，支持多种混合模式和透明度控制，常用于图像编辑和合成。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `base_image` | 必选 | IMAGE | - | - | 基础图像，作为粘贴的背景 |
| `cropped_image` | 必选 | IMAGE | - | - | 要粘贴的裁剪图像 |
| `bbox_meta` | 必选 | DICT | - | - | 边界框元数据，指定粘贴位置 |
| `blend_mode` | - | COMBO[STRING] | normal | normal, multiply, screen, overlay, soft_light, difference | 混合模式选择 |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | 不透明度，控制粘贴图像的透明程度 |
| `cropped_mask` | 可选 | MASK | - | - | 可选的遮罩，用于精确控制粘贴区域 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 粘贴完成后的合成图像 |

## 功能说明

### 混合模式
- **normal**：正常模式，直接覆盖原图像
- **multiply**：正片叠底，颜色相乘产生更暗的效果
- **screen**：滤色模式，产生更亮的效果
- **overlay**：叠加模式，结合multiply和screen的效果
- **soft_light**：柔光模式，产生柔和的光照效果
- **difference**：差值模式，计算颜色差异