# Load PS - 读取 PSD/PSB 图层

**节点功能：** `Load PS` 用于从单个 PSD/PSB 文件读取指定图层、全部图层批次或完整合成图，并输出对应的 alpha 遮罩、文件名和图层名称。节点支持拖拽/选择上传 PSD 文件，并在节点上显示当前模式的预览。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `file` | - | STRING | `""` | - | PSD/PSB 文件路径；支持 ComfyUI input 目录相对路径或绝对路径。 |
| `index` | - | INT | `0` | -8192-8192 | `output_mode=single_layer` 时使用的图层/组索引；支持负数索引（按数量取模）。 |
| `include_hidden` | - | BOOLEAN | `false` | - | 是否包含 Photoshop 中关闭眼睛的隐藏图层或隐藏组。 |
| `group_mode` | - | COMBO | `layer` | `layer`, `merged` | `layer` 展开组内图层；`merged` 将组作为一张合成图参与索引或批次。 |
| `output_mode` | - | COMBO | `all_layers` | `single_layer`, `all_layers`, `merged` | 输出指定单层、全部图层批次或完整 PSD 合成图。 |
| `preview` | - | BOOLEAN | `false` | - | 是否在节点上显示预览；关闭时拖拽上传只填入 `file`，不会触发 PSD 合成预览。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 读取得到的 RGBA 图像；`all_layers` 时为图层批次，其他模式为单张。 |
| `mask` | MASK | 与 `image` 对齐的 alpha 遮罩，可见区域为 1，透明区域为 0。 |
| `filename` | STRING | PSD/PSB 文件名（不含扩展名）。 |
| `layer_name` | STRING | 当前图层/组名称；批次模式下按换行分隔，顺序与 batch 对应。 |

## 功能说明

- 单文件输入：节点只读取一个明确的 PSD/PSB 文件，不扫描目录。
- 拖拽上传：可将 `.psd` / `.psb` 文件拖到节点上，或使用 `choose psd to upload` 按钮选择文件。
- 节点预览：开启 `preview` 后，`merged` 和 `single_layer` 显示单张预览，`all_layers` 显示图层网格预览。
- 图层模式：`single_layer` 按 `index` 选择一个图层/组，`all_layers` 输出全部有效图层/组为批次。
- 合成模式：`merged` 输出整个 PSD 的合成图，忽略 `index` 和 `group_mode`。
- 组处理：`group_mode=layer` 会展开组内图层；`group_mode=merged` 会把组作为一张图输出。
- 空内容过滤：空图层和空组合成结果会被过滤，不进入输出。

## 典型用法

- 将已栅格化的 PSD 图层读取为 IMAGE 批次，在 ComfyUI 中逐层处理。
- 使用 `single_layer` 和 `index` 单独读取某个图层或组。
- 使用 `merged` 快速获取 PSD 完整预览图。

## 注意与建议

- 该节点依赖 `psd-tools`，请确保已安装依赖。
- 建议输入已栅格化图层的 PSD；复杂图层样式、智能对象、调整图层等效果以 PSD/解析库可读取的结果为准。
- 大尺寸或多图层 PSD 建议保持 `preview=false`，需要查看时再临时开启。
