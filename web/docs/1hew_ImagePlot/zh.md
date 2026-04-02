# Image Plot - 图像布局拼接

**节点功能**：`Image Plot` 支持将图像水平、垂直或网格化排列。兼容列表格式的视频收集输入，构建跨批次的逐帧拼接视图。输入含 alpha 时会按 RGBA 处理并自动保留 alpha 输出。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE 或 LIST | - | - | 输入图像批次；或多个批次组成的列表，用于视频收集显示 |
| `layout` | - | COMBO | `horizontal` | `horizontal`/`vertical`/`grid` | 拼接布局模式 |
| `spacing` | - | INT | 10 | 0–1000 | 图像之间的间距 |
| `grid_columns` | - | INT | 2 | 1–100 | 网格模式下的列数 |
| `background_color` | - | STRING | `1.0` | 灰度/HEX/RGB；RGBA 输出时额外支持 `R,G,B,A` 与 `#RRGGBBAA` | 画布与 `spacing` 区域的背景色；为空时在 RGBA 输出下表示透明背景 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 合成图像批次；RGB 输入输出 RGB，RGBA 输入或含 alpha 的视频收集输入会自动保留 RGBA |

## 功能说明

- 标准拼接：将批次帧转换为 PIL，按 `layout` 排列后返回单张合成图像。
- 视频收集：接受批次列表输入，跨组进行帧对齐的并列拼接，保留设备与数据类型。
- 尺寸归一：对输入图片进行双线性至最小公共尺寸，确保对齐整齐。
- 颜色解析：支持灰度值（`0.0–1.0`）、HEX（`#RRGGBB`）与整数 `R,G,B`。
- Alpha 规则：只要输入批次或视频收集中的任意图像带 alpha，画布就会按 RGBA 创建并保留透明度。
- 背景规则：`background_color` 同时控制画布和 `spacing` 区域。
- 透明规则：仅当 `background_color` 为空或显式写成 `transparent` / `none` 等值时，RGBA 输出才使用透明背景。
- 颜色规则：RGBA 输出下，普通灰度/HEX/RGB 颜色会按对应颜色的不透明背景填充；如需自定义透明度，可传 `R,G,B,A` 或 `#RRGGBBAA`。
- 原图保护：`background_color` 只填充非图像矩形区域，不会覆盖原图内部原本就存在的透明区域。

## 典型用法

- 并排对比：设置 `layout=horizontal` 并调整 `spacing`。
- 垂直堆叠：设置 `layout=vertical`，形成列式视图。
- 网格展示：设置 `layout=grid` 并调 `grid_columns` 实现平铺。
- 多序列显示：传入批次列表，逐帧拼接各输入以观察时间一致性。

## 注意与建议

- 视频收集判定基于 Python 列表输入；张量输入视为单个批次。
- 输出为末通道张量并限制在 `[0,1]` 范围。