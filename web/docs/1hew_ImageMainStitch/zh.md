# Image Main Stitch - 主画面拼接

**节点功能：** `Image Main Stitch` 节点以 `image_1` 为主画面锚点，将 `image_2..image_N` 先拼接成组合，再按 `direction` 与主画面合并，并可添加间隔条。输出的 `mask` 将 `image_1` 区域标记为 `1`，组合与间隔区域标记为 `0`。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image_1` | 必需 | IMAGE | - | - | 主画面图像批次，作为拼接锚点。 |
| `image_2` | 可选 | IMAGE | - | - | 追加图像批次，参与组合拼接。 |
| `image_3` | 可选 | IMAGE | - | - | 追加图像批次，参与组合拼接。 |
| `image_4..image_N` | 可选 | IMAGE | - | - | 动态追加输入，按编号顺序收集并拼接。 |
| `direction` | - | COMBO | `left` | `top` / `bottom` / `left` / `right` | 组合相对 `image_1` 的摆放方向。 |
| `match_image_size` | - | BOOLEAN | True | - | 为 True 时按等比缩放匹配拼接边长；为 False 时按居中填充对齐尺寸。 |
| `spacing_width` | - | INT | 10 | 0–1000 | 主画面与组合之间、组合内部图像之间的间隔条宽度/高度。 |
| `spacing_color` | - | STRING | `1.0` | 灰度/HEX/RGB | 间隔条颜色。 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/`edge`/`average`/`extend`/`mirror` | 尺寸对齐时的填充策略。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 拼接后的图像批次。 |
| `mask` | MASK | 区域遮罩：`image_1` 区域为 `1`，其余区域为 `0`。 |

## 功能说明

- 动态输入：支持 `image_2..image_N`，并按编号从小到大顺序拼接。
- 两阶段布局：先构建组合，再按 `direction` 与主画面合并。
- 批次广播：输入批次大小自动扩展到最大批次，便于批处理。
- 尺寸对齐：
  - `match_image_size=True`：使用双三次插值等比缩放并保持比例。
  - `match_image_size=False`：使用 `pad_color` 进行居中填充对齐尺寸。
- 颜色解析：
  - `spacing_color`：支持灰度、HEX、RGB 与常见颜色名。
  - `pad_color`：在此基础上支持 `edge`、`average`、`extend`、`mirror` 策略。

## 典型用法

- 主画面配参考条：使用 `direction=left/right` 将竖向参考组合贴到主画面侧边。
- 顶部/底部对比栏：使用 `direction=top/bottom` 将横向组合贴到主画面上方或下方。
- 主区域遮罩输出：将 `mask` 作为选择器，配合后续节点仅对主画面区域执行处理。

## 注意与建议

- 仅连接 `image_1` 时，节点输出 `image_1` 与全白 `mask`。
- `spacing_width=0` 时区域之间直接贴合。
