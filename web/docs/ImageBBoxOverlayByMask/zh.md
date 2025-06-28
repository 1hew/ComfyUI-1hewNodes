# Image BBox Overlay by Mask - 基于遮罩的图像边界框叠加

**节点功能：** `Image BBox Overlay by Mask`节点根据遮罩生成检测框并以描边形式叠加到图像上，支持独立模式和合并模式，可为每个独立的遮罩区域生成单独的边界框或将所有遮罩合并为一个边界框。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要叠加边界框的图像 |
| `mask` | 必选 | MASK | - | - | 用于生成边界框的遮罩 |
| `bbox_color` | - | COMBO[STRING] | red | red, green, blue, yellow, cyan, magenta, white, black | 边界框颜色 |
| `line_width` | - | INT | 3 | 1-20 | 边界框线条宽度 |
| `padding` | - | INT | 0 | 0-50 | 边界框填充像素数 |
| `output_mode` | - | COMBO[STRING] | separate | separate, merge | 输出模式：separate（独立模式），merge（合并模式） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 叠加边界框后的图像 |