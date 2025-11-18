# Multi Image Stitch - 方向拼接与间距控制

**节点功能：** `Multi Image Stitch` 在指定方向（`top`/`bottom`/`left`/`right`）拼接多张图像，并可配置间距宽度与颜色。支持“尺寸匹配（按比例缩放）”与“补边匹配”，并安全处理批次。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `direction` | - | COMBO | `right` | `top` / `bottom` / `left` / `right` | 拼接方向 |
| `match_image_size` | - | BOOLEAN | True | - | `True` 时按比例缩放匹配；否则按 `pad_color` 补边统一 |
| `spacing_width` | - | INT | 10 | 0–1000 | 图像间的间距宽度 |
| `spacing_color` | - | STRING | `1.0` | 灰度/HEX/RGB/颜色名 | 间距颜色 |
| `pad_color` | - | STRING | `1.0` | 补边策略或颜色 | 补边策略或颜色 |
| `image_1` | - | IMAGE | - | - | 第一张图像 |
| `image_2…image_N` | 可选 | IMAGE | - | - | 额外图像，按数字后缀排序识别 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 拼接后的图像批次；裁剪到 `[0,1]` 浮点 |

## 功能说明

- 迭代拼接：按顺序将 `image_1..N` 逐对拼接。
- 批次广播：自动对齐不同批次数量。
- 两种匹配：
- 比例缩放匹配：`match_image_size=True`。
- 补边统一：`match_image_size=False`，使用 `pad_color` 策略。
- 间距条：按方向生成纵/横条，颜色由 `spacing_color` 决定；`spacing_width=0` 时无间距。
- 高级补边：支持 `extend`、`mirror`、`edge`、`average` 或显式颜色。

## 典型用法

- 拼贴与海报：在横/竖方向拼接成条幅或栏目，并控制间距。
- 布局一致性：用比例缩放保持形状一致；用补边保留完整内容。
- 批次安全：对不同批次数量的图像进行自动广播后拼接。

## 注意与建议

- `spacing_color` 支持灰度浮点、`R,G,B`（`0..1` 或 `0..255`）、`#hex` 与常见颜色名。
- 左/右方向统一高度，顶/底方向统一宽度以保证对齐。