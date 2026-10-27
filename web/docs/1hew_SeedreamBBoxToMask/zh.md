# Seedream BBox to Mask - Seedream bbox 转遮罩

**节点功能：** `Seedream BBox to Mask` 把 Seedream 的 bbox 还原成矩形遮罩，只处理 bbox：
prompt 里的 `<bbox>x1 y1 x2 y2</bbox>`，以及图层拆分返回的 `bounding_box`。
`<point>x y</point>` 这类点标记不属于本节点，会被忽略。

坐标只有**两种单位**，由 `coord_mode` 明确指定，不做自动猜测：

- `normalized`（默认）：千分位 0-1000，正向换算 `x = round(x_px / width * 1000)`，
  反解 `x_px = round(x / 1000 * width)`。对应 prompt 标记与 `bounding_box.normalized`。
- `absolute`：像素，对应 `bounding_box.absolute`。

两种单位都是 `[left, top, right, bottom]`（xyxy），`right` / `bottom` 是边界而不是宽高，
所以没有 box_format 选项。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 参考图像，只用来提供输出 mask 的宽高；批次内尺寸一致 |
| `coordinates_string` | - | STRING | 空 | 多行 | 每行一条，格式固定 `xyxy`；支持 `<bbox>` 标签、`bbox(...)`、JSON 数组/对象、带标签的行（`xyxy: 1 2 3 4`）与裸数字；行首 `#` / `//` / `--` 为整行注释 |
| `coord_mode` | - | COMBO | `normalized` | `normalized` / `absolute` | 坐标单位：千分位 0-1000 或像素 |
| `output_mode` | - | COMBO | `separate` | `separate` / `merge` | `separate`：每个框一张 mask（批次）；`merge`：全部合并成一张 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `mask` | MASK | `separate`：`N×H×W`（N 为解析到的框数量）；`merge`：`1×H×W`；无框时 `separate` 返回 `0×H×W`、`merge` 返回全零 |
| `count` | INT | 解析到的框数量 |

## 功能说明

- 反解公式与官方一致：`px = round(n / 1000 * 边长)`，越界坐标按图像尺寸收敛。
- 解析优先级：标签 → 只有开标签 → 函数式 `bbox(...)` → JSON 数组/对象 → 每行裸数字。
- 带标签时只取标签内的数字，所以 `2 图1<bbox>120 180 640 760</bbox> 区域替换成花园`
  这种整句提示词只会得到 1 个框，行号与 `图1`、`Image 1` 前缀都不会被当成坐标。
- 图层拆分：`bounding_box` 里同时有 `absolute` 和 `normalized` 时按 `coord_mode` 取对应字段；
  只剩一个字段时直接用该字段自己的单位换算，所以两种 JSON 都能整段贴进来。
- `normalized` 模式下如果出现 >1000 的值，会直接报错提示改用 `absolute`，不会静默算出错误遮罩。
- 注释只给人看：行首 `#` / `//` / `--` 开头的整行会被丢弃，注释里出现的 `<bbox>` 或数字**不会**被解析。
- 单位不要写在注释里：`coord_mode` 才是唯一依据，注释写错不会改变结果（也不会报错）。
- 全角数字与全角标点会自动转半角。

## 典型用法

- 官方示例：`<bbox>120 180 640 760</bbox>` 在 640×480 图上得到 `[77, 86, 410, 365]`。
- 多框：每行一条，`separate` 输出批量 mask，可直接接逐框处理或局部回贴。
- 框选预览：把 `merge` 的 mask 叠到原图上确认范围。
- 自我说明：坐标块里可以直接写注释，例如

  ```text
  # Seedream bbox, xyxy, normalized 0-1000
  120 180 640 760
  # 第二个框（像素时把 coord_mode 改成 absolute）
  500 400 999 999
  ```
- 图层还原：整段拆分 JSON 贴进来即可；`absolute` 无取整误差，`normalized` 换任意画布更通用
  （官方提示 ±1px 取整误差）。

## 注意与建议

- 只解析 bbox：`<point>` 及只有两个数字的行会被忽略（`count` 不增加）。
- 图层拆分场景要接**输出底图**的尺寸（如 2048×2048），不是图层 PNG 自己的 `size`（如 1273×265）；
  `bounding_box` 的坐标系永远是输出底图。
- Seedream 5.0 pro 的编辑接口只接受 `image` 与 `prompt`，没有 mask 入参；本节点产出的 mask
  用于本地回贴、预览或其它模型，不会作为 API 参数上传。
- 同一批次内的图像尺寸必须一致，遮罩沿用第一帧的宽高。
