# Save Text - 保存文本到磁盘

**节点功能：** `Save Text` 将输入的文本写入指定格式的文件（`txt` / `csv` / `md` / `json`），同时返回保存后的绝对路径。支持相对/绝对路径、自动编号、覆盖保存、`format` 格式选择，以及仅预览不落盘的模式。文件固定使用 UTF-8 编码（与 ComfyUI 原生 Save Text 一致）。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `text` | 可选 | STRING | - | - | 要保存的文本；不连接时为空（None），节点预览会同步清空。 |
| `filename` | - | STRING | `text/ComfyUI` | - | 保存文件前缀或绝对路径。相对路径会写入 ComfyUI 输出目录下。 |
| `format` | - | COMBO | `txt` | `txt` / `csv` / `md` / `json` | 输出文件格式。`json` 在内容合法时会缩进格式化；csv/json 不追加元数据。 |
| `auto_increment` | - | BOOLEAN | `true` | - | 为 `true` 时自动编号生成新文件；为 `false` 时使用固定文件名并覆盖。 |
| `save_output` | - | BOOLEAN | `true` | - | 为 `true` 时写入磁盘；为 `false` 时仅在 UI 中预览文本，不保存文件。 |
| `save_metadata` | - | BOOLEAN | `true` | - | 为 `true` 时在文本/Markdown 文件末尾追加 workflow/prompt 元数据，拖动该文件到 ComfyUI 可复现工作流；csv/json 会自动跳过。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `file_path` | STRING | 保存后的绝对路径；当 `save_output=false` 时返回空字符串。 |

## 功能说明

- 文本输入：接收 STRING 输入；不连接时按 `None` 处理，并清空节点内预览。
- 路径兼容：支持相对路径和绝对路径；相对路径默认保存到输出目录下。
- 自动编号：开启 `auto_increment` 时按 `_00001_` 形式递增，避免覆盖已有文件。
- 覆盖模式：关闭 `auto_increment` 时固定写入 `filename.<format>`。
- 格式支持：`format` 可选 `txt` / `csv` / `md` / `json`，默认 `txt`；扩展名随格式变化。
- JSON 格式化：`json` 格式会尝试解析内容并缩进输出；内容不是合法 JSON 时按原文写入。
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
- `format` 位于 `filename` 下方。`encode` 已移除（固定 UTF-8）；加载旧工作流时前端会自动迁移旧的 `widgets_values` 顺序，缺少 `format` 时默认 `txt`，无需手动重设参数。
- 输入端口已从 `any` 更名为 `text`（类型 STRING）；旧工作流需要把连线重新接到 `text`。
