# ImageMainStitch - 主画面拼接

**节点功能**：`ImageMainStitch` 先将 `image_2..image_N` 顺序拼成组合（按方向选择水平或垂直），随后将该组合贴到 `image_1` 的指定侧。支持统一的 `spacing_width` 与尺寸匹配/填充策略。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image_1` | - | IMAGE | - | - | 主图像批次 |
| `image_2` | - | IMAGE | - | - | 次图像批次 |
| `image_3` | - | IMAGE | - | - | 三图像批次 |
| `direction` | - | COMBO | `left` | `top`/`bottom`/`left`/`right` | 将组合贴到 `image_1` 的哪一侧 |
| `match_image_size` | - | BOOLEAN | True | - | `True` 沿目标轴等比缩放；`False` 通过居中填充统一尺寸，不改变原图 |
| `spacing_width` | - | INT | 10 | 0–1000 | 同时用于 2+3 合并与最终贴合的间隔宽度 |
| `spacing_color` | - | STRING | `1.0` | 灰度/HEX/RGB | 间隔条颜色 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/`edge`/`average`/`extend`/`mirror` | 尺寸统一时的填充策略 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 拼接后的图像批次 |

## 功能说明

- 两阶段合成：`top/bottom` 下 2 与 3 先水平并排，再与 1 垂直拼接；`left/right` 下先垂直堆叠，再与 1 水平拼接。
- 尺寸统一：`match_image_size=True` 沿目标轴等比缩放；否则进行居中填充。