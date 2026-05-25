# Image Pad By BBox Mask - 按边界遮罩填充画布

**节点功能：** `Image Pad By BBox Mask` 根据 `bbox_mask` 的白色区域外接框，将 `paste_image` 等比缩放并居中放入该区域，输出与 `bbox_mask` 同尺寸的图像；未被图像覆盖的区域使用 `pad_color` 填充。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `paste_image` | - | IMAGE | - | - | 需要放入 bbox 区域的图像；RGBA 输入会保留 alpha 通道 |
| `bbox_mask` | - | MASK | - | - | 用于计算输出尺寸与白色区域外接框的遮罩 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/颜色名/`edge`/`average`/`extend`/`mirror` | 非图像区域的填充颜色或填充策略 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 按 bbox 放置并填充后的图像 |

## 功能说明

- 边界框检测：沿用 `Image Paste By BBox Mask` 的 `bbox_mask` 逻辑，从白色区域提取外接矩形。
- 等比放置：`paste_image` 保持原比例 fit 到 bbox 内，并在 bbox 中居中。
- 画布尺寸：输出宽高来自 `bbox_mask`，适合把局部图像重新铺回原遮罩坐标空间。
- 填充策略：支持纯色、灰度、RGB、十六进制、颜色名，以及 `edge`、`average`、`extend`、`mirror`。
- 批量处理：`paste_image` 与 `bbox_mask` 批次数不一致时按取模方式循环匹配。

## 典型用法

- 将裁剪后的局部图像补回原始遮罩画布，生成和原图坐标一致的 padded image。
- 为局部图像补边后送入后续图像编辑、合成或对齐节点。
- 配合 `Mask To BBox Mask` 先生成矩形 bbox，再用本节点恢复局部图像在整张画布中的位置。

## 注意与建议

- `bbox_mask` 只用于计算外接矩形，不按任意形状逐像素混合。
- 当 `bbox_mask` 为空时，节点会输出一张与遮罩同尺寸的 `pad_color` 背景图。
- 如果 `paste_image` 与 bbox 比例不同，bbox 内未覆盖的边缘也会使用 `pad_color` 填充。
