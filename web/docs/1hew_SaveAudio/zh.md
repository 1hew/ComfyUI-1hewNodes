# Save Audio - 保存音频到磁盘并写入元数据

**节点功能：** `Save Audio` 将 AUDIO 输入保存到 ComfyUI 输出目录或临时目录，并返回保存后的绝对文件路径。支持 FLAC / WAV / MP3 / Opus 四种格式；`audio` 输入为可选，未连接（或内容为空）时节点直接空输出，不写入任何文件。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `audio` | 可选 | AUDIO | - | - | 要保存的音频（`waveform` 形如 `[batch, channels, samples]`，附带 `sample_rate`）；为空时节点输出空结果。 |
| `filename` | - | STRING | `audio/ComfyUI` | - | 保存文件名或路径前缀，交由 ComfyUI 路径生成处理；支持日期占位符（如 `%date:yyyy-MM-dd%`）。 |
| `format` | - | COMBO | `flac` | `flac` / `wav` / `mp3` / `opus` | 保存格式；mp3 / opus 在采样率不受编码器支持时会自动重采样。 |
| `auto_increment` | - | BOOLEAN | `true` | - | 为 `true` 时自动编号生成新文件；为 `false` 时使用固定文件名并覆盖。 |
| `save_output` | - | BOOLEAN | `true` | - | 开启时保存到输出目录；未开启时保存到临时目录。 |
| `save_metadata` | - | BOOLEAN | `true` | - | 开启时将 prompt/workflow 元数据写入音频容器。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `file_path` | STRING | 保存后音频文件的绝对路径；批量音频以换行分隔。 |

## 功能说明

- 空输入直通：`audio` 未连接、为 `None`，或 `waveform` / `sample_rate` 为空时直接返回空输出，不创建任何文件；同时会向前端发送空的音频列表，把上一次运行留下的播放器清空。
- 批量拆分：`waveform` 的 batch 维度会逐条保存为独立文件；固定文件名模式下从第二个文件起追加 `_001`、`_002` 后缀。
- 声道支持：单声道与双声道；更多声道会给出明确错误而不是写出损坏文件。
- 采样率适配：flac / wav 原样保存；mp3 / opus 会就近选择编码器支持的采样率并重采样。
- 元数据：可选把 prompt/workflow 写入容器 metadata；ComfyUI 以 `--disable-metadata` 启动时自动跳过。
- 失败清理：单个文件编码失败会删除半成品并抛出带文件名的错误。
- 节点内播放器：前端扩展会在节点里嵌入一个音频播放器，执行完成后自动指向刚保存的文件（`/view?filename=...&type=output`），可直接试听，与原生 `Save Audio (Advanced)` 的预览一致；没有产出音频时播放器自动隐藏。
- UI 数据：同时返回 `SavedAudios`（旧版本回退为 `PreviewAudio`），供前端读取音频文件名、子目录与目录类型。

## 典型用法

- 保存音频生成节点（TTS、音乐、视频音轨）的结果，用于归档或交付。
- 只想试听时关闭 `save_output`，文件写入临时目录，避免污染输出目录，节点内的播放器依然可以播放。
- 批量音频一次保存为多个文件，保持与输入 batch 一一对应。

## 注意与建议

- 节点依赖 `PyAV`（ComfyUI 自带）编码；若环境缺少某个编码器（如 `libopus`），该格式会报错，可改用 flac / wav。
- `auto_increment=false` 会覆盖同名文件，批量保存时请确认目标目录。
- 绝对路径前缀会跳过输出目录直接写入指定位置；若保存的文件不在 ComfyUI 的静态目录下，预览会回退为 `PreviewAudio(audio)`，播放的是输入音频而不是已保存文件（旧版本可能没有预览）。
