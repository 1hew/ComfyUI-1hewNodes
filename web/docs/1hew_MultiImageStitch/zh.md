# Multi Image Stitch - 多图像缝合

**节点功能：** `Multi Image Stitch` 节点按指定方向对动态 `image_X` 输入进行顺序缝合，支持配置间距宽度与颜色、可选统一画布尺寸，以及居中对齐。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image_1` | 可选 | IMAGE | - | - | 动态图像输入首端口；可扩展 `image_2`、`image_3` 等 |
| `direction` | 必选 | COMBO[STRING] | right | top, bottom, left, right | 缝合方向 |
| `match_image_size` | 必选 | BOOLEAN | True | True/False | 是否在缝合前统一图像画布尺寸 |
| `spacing_width` | 必选 | INT | 10 | 0–1000 | 相邻图像之间的间距宽度 |
| `spacing_color` | 必选 | STRING | 1.0 | 颜色字符串 | 间距区域的填充颜色 |
| `pad_color` | 必选 | STRING | 1.0 | 颜色字符串 | 对齐时用于画布填充的颜色 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 按方向、间距与对齐规则缝合后的图像 |
