# Mask To Image - 将遮罩映射为 RGB / RGBA 图像

**节点功能：** `Mask To Image` 将输入的 `MASK` 直接映射为图像输出。可分别设置 mask 白色区域和黑色区域对应的颜色，并支持输出带 alpha 的 RGBA 图像。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `mask` | - | MASK | - | - | 输入遮罩，支持灰度过渡 |
| `fill_hole` | - | BOOLEAN | `False` | `True / False` | 为 `True` 时先填充 mask 内部封闭黑洞，再执行颜色映射 |
| `white_area_color` | - | STRING | `1.0` | 与 `Image Solid` 的 `color` 相同：颜色名 / `#RRGGBB` / `r,g,b` / `0~1` 灰度 / 单字母颜色简写 | mask 白色区域映射到的颜色 |
| `black_area_color` | - | STRING | `0.0` | 与 `Image Solid` 的 `color` 相同：颜色名 / `#RRGGBB` / `r,g,b` / `0~1` 灰度 / 单字母颜色简写 | mask 黑色区域映射到的颜色 |
| `output_alpha` | - | BOOLEAN | `False` | `True / False` | 为 `True` 时输出 RGBA，并将原始 mask 作为 alpha 通道；RGB 颜色映射保持不变 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 映射后的 RGB 或 RGBA 图像 |

## 功能说明

- 灰度保留：mask 中间灰度会线性映射，不会被强制二值化。
- RGB 模式：按 `black_area_color -> white_area_color` 的渐变关系输出彩色图。
- RGBA 模式：RGB 仍按 `black_area_color -> white_area_color` 映射，同时将原始 mask 灰度写入 alpha 通道。
- fill_hole：在颜色映射前先执行孔洞填充，适合封闭区域补满。
- 颜色输入兼容：与 `Image Solid` 的 `color` 输入规则保持一致。

## 典型用法

- 把灰度 mask 转为可视化彩色图。
- 生成仅保留前景颜色、背景透明的 RGBA 蒙版图层。
- 为下游 API 或图像编辑节点准备彩色 brush / matte 输入。

## 注意与建议

- `output_alpha=true` 时，透明度来自原始 mask，而不是额外阈值处理。
- 若想得到纯黑底彩色图，请关闭 `output_alpha` 并把 `black_area_color` 设为 `0.0`。
- 若想得到带透明通道的结果，请开启 `output_alpha`；此时 `white_area_color` 和 `black_area_color` 的 RGB 映射仍然生效。
