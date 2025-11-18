# Image Edit Stitch - 参考图与编辑图拼接

**节点功能**：`Image Edit Stitch` 将参考图与编辑图按指定方向拼接，可设置分隔条。`match_edit_size=False` 时保留参考图原始比例；自动将编辑遮罩与编辑图对齐；同时输出合成遮罩与“编辑/参考”分离遮罩。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `reference_image` | - | IMAGE | - | - | 参考图像批次 |
| `edit_image` | - | IMAGE | - | - | 编辑图像批次 |
| `edit_mask` | 可选 | MASK | - | - | 与编辑图像对齐的遮罩；未提供时自动创建全白遮罩 |
| `edit_image_position` | - | COMBO | `right` | `top`/`bottom`/`left`/`right` | 编辑图相对于参考图的放置侧 |
| `match_edit_size` | - | BOOLEAN | False | - | `True` 时参考图按目标边带填充对齐；`False` 时保留参考图比例 |
| `spacing` | - | INT | 0 | 0–1000 | 图像间的分隔条宽/高 |
| `spacing_color` | - | STRING | `1.0` | 灰度/HEX/RGB | 分隔条颜色（严格 RGB 0..1） |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/`edge`/`average`/`extend`/`mirror` | 尺寸对齐的填充策略 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 拼接后的图像批次 |
| `mask` | MASK | 合成遮罩；编辑区域保留，其余为 0 |
| `split_mask` | MASK | 分离遮罩；编辑区域为 1，参考区域为 0；分隔条区域为 0 |

## 功能说明

- 比例处理：`match_edit_size=False` 保持参考图纵横比；为匹配尺寸时按目标边居中填充。
- 遮罩对齐：编辑遮罩使用最近邻与编辑图严格对齐，保持二值特性。
- 颜色解析：填充支持 `edge`/`average`/`extend`/`mirror`；分隔条颜色采用严格 RGB。
- 批量广播：自动将参考/编辑/遮罩广播到统一的最大批次长度。
- 方向变体：`top`/`bottom`/`left`/`right` 均提供一致的分隔与遮罩合成逻辑。

## 典型用法

- A/B 对比：设置 `edit_image_position=right`、`spacing>0`，用于编辑前后对照。
- 垂直堆叠：使用 `top/bottom` 进行上下拼接，通过 `spacing_color` 提供分隔视觉。
- 保留参考风格：设置 `match_edit_size=False` 以保持参考图比例与观感。

## 注意与建议

- 当仅提供一侧图像时，节点会返回该图像与全白遮罩，并输出语义分离遮罩以标识区域。
- 分隔条以常量颜色张量按批次扩展，计算高效。