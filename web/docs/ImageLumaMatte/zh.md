# Image Luma Matte - 亮度蒙版

**节点功能：** `Image Luma Matte`节点通过将遮罩应用到图像上创建基于亮度的合成效果，支持批量处理、羽化边缘和可自定义的背景选项，支持多种颜色格式和特殊值。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要处理的输入图像 |
| `mask` | 必选 | MASK | - | - | 定义蒙版区域的遮罩 |
| `invert_mask` | 可选 | BOOLEAN | False | True/False | 是否反转遮罩 |
| `feather` | 可选 | INT | 0 | 0-50 | 羽化半径，用于软化遮罩边缘 |
| `background_add` | 可选 | BOOLEAN | True | True/False | 是否添加背景或创建透明输出 |
| `background_color` | 可选 | STRING | "1.0" | 多种格式 | 背景颜色，支持多种格式和特殊值 |
| `out_alpha` | 可选 | BOOLEAN | False | True/False | 是否输出RGBA格式（包含alpha通道） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 应用亮度蒙版后的处理图像 |