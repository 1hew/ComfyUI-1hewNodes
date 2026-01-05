# Save Video by Image - 序列帧编码为视频

**节点功能：** `Save Video by Image` 用于将 IMAGE 批次按指定 FPS 编码保存为视频文件。支持可选音频混流，并对包含 Alpha 的图像提供兼容的输出策略。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 作为视频帧的图像批次。 |
| `audio` | 可选 | AUDIO | - | - | 可选音频，用于混流到输出视频。 |
| `fps` | - | FLOAT | `8.0` | 0.01-120 | 编码帧率（每秒帧数）。 |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | 文件名前缀；遵循 ComfyUI 的保存路径规则。 |
| `save_output` | - | BOOLEAN | `true` | - | 保存到输出目录（true）或临时目录（false）。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `file_path` | STRING | 保存后视频文件的绝对路径。 |

## 功能说明

- FFmpeg 编码：以 rawvideo 方式向 ffmpeg 流式写入帧并完成编码。
- 音频混流：将输入音频保存为临时 WAV，并在编码时进行混流。
- Alpha 输出策略：
  - RGBA 且 `save_output=true`：输出 `.mov`（ProRes，保留 Alpha），并在临时目录生成 `.webm` 预览。
  - RGBA 且 `save_output=false`：输出带 Alpha 的 `.webm` 到临时目录。
  - RGB：输出 `.mp4`（H.264）到目标目录。
- 尺寸对齐：对常见编码器要求的偶数宽高进行自动调整。

## 典型用法

- 将处理后的帧批次快速编码为可预览视频，并将输出路径交给后续节点使用。

## 注意与建议

- 编码依赖 `ffmpeg` 可执行文件处于可访问环境中。

