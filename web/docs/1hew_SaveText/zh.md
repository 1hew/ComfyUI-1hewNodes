# Save Text - 保存文本到磁盘

**节点功能：** `Save Text` 将输入写入指定格式的文件（`txt` / `csv` / `md` / `json`），同时返回保存后的绝对路径。`text` 为通配输入，除了字符串，也可以直接接 JSON 安全的字典、列表、数字与布尔。支持相对/绝对路径、自动编号、覆盖保存、`format` 格式选择，以及仅预览不落盘的模式。文件固定使用 UTF-8 编码（与 ComfyUI 原生 Save Text 一致）。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `text` | 可选 | * | - | - | 要保存的内容：`str`、`bytes`、`dict`、`list`、`int`、`float`、`bool` 或 `None`。容器会按 JSON 文本写入；`csv` 仍只接受文本。不连接时为空（None），节点预览会同步清空。 |
| `filename` | - | STRING | `text/ComfyUI` | - | 保存文件前缀或绝对路径。相对路径会写入 ComfyUI 输出目录下。 |
| `format` | - | COMBO | `txt` | `txt` / `csv` / `md` / `json` | 输出文件格式。`json` 在内容合法时会缩进格式化；csv/json 不追加元数据。 |
| `auto_increment` | - | BOOLEAN | `true` | - | 为 `true` 时自动编号生成新文件；为 `false` 时使用固定文件名并覆盖。 |
| `save_output` | - | BOOLEAN | `true` | - | 为 `true` 时写入磁盘；为 `false` 时仅在 UI 中预览文本，不保存文件。 |
| `save_metadata` | - | BOOLEAN | `true` | - | 为 `true` 时在文本/Markdown 文件末尾追加 workflow/prompt 元数据，拖动该文件到 ComfyUI 可复现工作流；csv/json 会自动跳过。 |

## 输入类型与格式策略

| 输入 \ format | `txt` / `md` | `json` | `csv` |
| --- | --- | --- | --- |
| `str` | 原样 | 合法 JSON 则缩进美化，否则原样 | 原样 |
| `bytes` | UTF-8 解码后按 `str` | 同左 | 同左 |
| `dict` / `list` / `tuple` | 美化 JSON 文本 | 美化 JSON 文本 | 报错：csv 只接受文本 |
| `int` / `float` | 纯文本 | JSON 数字（`NaN` / `Infinity` 报错） | 纯文本 |
| `bool` | `True` / `False` | `true` / `false` | `True` / `False` |
| `None`（未连接） | 清空预览、空输出 | 同左 | 同左 |
| tensor / 音频 / 其它 | 报错 | 报错 | 报错 |

对象的键必须是字符串。JSON 文档的根可以是任意 JSON 值（对象、数组、字符串、数字、布尔）；很多下游 API 习惯根为对象或数组，需要时用 `Dict` 包一层。set、容器内的 bytes、tensor、`NaN` / `Infinity` 会带路径报错，不会被悄悄转成字符串。顶层 `None` 仍表示未连接。

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `file_path` | STRING | 保存后的绝对路径；当 `save_output=false` 时返回空字符串。 |

## 功能说明

- 通配输入：接收字符串、字典、列表、数字、布尔与 bytes；不连接时按 `None` 处理，并清空节点内预览。
- 路径兼容：支持相对路径和绝对路径；相对路径默认保存到输出目录下。
- 自动编号：开启 `auto_increment` 时按 `_00001_` 形式递增，避免覆盖已有文件。
- 覆盖模式：关闭 `auto_increment` 时固定写入 `filename.<format>`。
- 格式支持：`format` 可选 `txt` / `csv` / `md` / `json`，默认 `txt`；扩展名随格式变化。
- JSON 格式化：`json` 对字符串内容尝试解析并缩进，对 dict/list 始终输出缩进 JSON；非 JSON 值会直接报错。
- 预览模式：`save_output=false` 时不写磁盘，但仍可在 UI 面板查看文本内容。
- 工作流元数据：`save_metadata=true` 时，在文本/Markdown 文件正文后追加一个 HTML 注释形式的元数据块（包含 UI workflow 与 API prompt）；把生成的 `.txt` 直接拖到 ComfyUI 画布即可加载对应工作流，效果与拖入带元数据的 PNG/视频一致。csv/json 会跳过元数据以保持文件结构合法。
- 编码：固定 UTF-8（与原生 Save Text 一致），不再提供 `encode` 控件。元数据块只使用 ASCII 字符（非 ASCII 内容以 \uXXXX 转义），仍可被前端识别。
- 节点内预览：运行后在节点底部以只读文本面板显示输出内容，支持选中和滚动；无输入时预览会清空，不再显示上一次的内容。

## 典型用法

- 将提示词、日志或 JSON 结果保存为对应格式的文件，便于外部流程读取。
- 保存带元数据的提示词/结果文本文件，之后把该文件拖回 ComfyUI 画布，快速复现生成它的工作流。
- 在调试阶段关闭 `save_output`，仅查看节点实际输出文本。

## 注意与建议

- 绝对路径模式下请确保目标目录具有写入权限。
- 批量并发保存时，节点内部会串行分配文件路径，减少重名冲突。
- 开启 `save_metadata` 后，文本/Markdown 文件正文末尾会多出元数据注释块；需要交付纯净文本（如给外部脚本或 LLM 直接读取）时请关闭该选项。`Load Text` 读取时会自动剥离该块，仍返回原始正文。csv/json 格式下元数据会被自动跳过。
- 文本文件拖拽复现依赖本节点包自带的前端脚本；仅更新后端或禁用前端扩展时，ComfyUI 原生前端不会读取文本元数据。
- 节点 ID 为 `1hew_SaveText`（v4.5.0 由 `1hew_SaveTxt` 改名；显示名、端口、默认值与文件格式均未变）。扩展注册了 ComfyUI 节点替换：旧 ID 的工作流可在「缺失节点」面板一键换成新节点（连线与 widget 值会一并迁移），仍发送 `1hew_SaveTxt` 的 API prompt 会在执行前自动改写；不支持节点替换的旧版 ComfyUI 需手动重新添加节点。
- `format` 位于 `filename` 下方；`encode` 已移除（固定 UTF-8）。v4.0.0 之前保存的工作流可能还带旧的 `encode` 值，替换旧节点后请复核 `auto_increment` / `save_output` / `save_metadata`。
- 输入端口已从 `any` 更名为 `text`，现在是通配（`*`）输入，dict/list 可以直接接入；旧的 STRING 连线仍然有效，因为通配输入接受所有类型。
