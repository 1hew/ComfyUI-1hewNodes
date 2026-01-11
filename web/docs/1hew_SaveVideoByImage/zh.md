# Save Video by Image - 由图像批次编码视频

**节点功能：** `Save Video by Image` 将 IMAGE 图像批次按帧序编码为视频，并支持将 AUDIO 输入混流进输出文件。节点返回保存后的绝对文件路径，并提供 UI 预览入口。对包含透明通道的序列，节点以 WEBM（VP9）作为预览输出；在保存到输出目录时可同时导出高质量 MOV（ProRes）成品。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 图像批次按时间顺序作为视频帧进行编码。 |
| `audio` | 可选 | AUDIO | - | - | 音频输入；提供时混流进输出视频。 |
| `fps` | - | FLOAT | `8.0` | 0.01-120.0 | 编码帧率。 |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | 保存文件前缀，交由 ComfyUI 路径生成处理；通常支持日期占位符（如 `%date:yyyy-MM-dd%`）。 |
| `save_output` | - | BOOLEAN | `true` | - | 开启时保存到输出目录；未开启时保存到临时目录。 |
| `save_metadata` | - | BOOLEAN | `true` | - | 开启时将 prompt/workflow 元数据写入容器 `comment` 字段。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `file_path` | STRING | 保存后视频文件的绝对路径。 |

## 功能说明

- 帧序编码：通过 stdin 向 `ffmpeg` 流式写入 rawvideo 帧进行编码。
- 音频混流：将 AUDIO 转为临时 WAV 文件并混流进输出。
- 偶数尺寸对齐：对宽高做偶数对齐，编码过程更稳定。
- 透明通道输出策略：
  - 帧包含透明通道时，默认导出 WEBM（VP9 + yuva420p）。
  - 帧包含透明通道且 `save_output=true` 时，在临时目录生成预览 WEBM，
    同时在输出目录导出 MOV（ProRes 4444）成品。
- 元数据写入：将 prompt/workflow JSON 写入 `comment` 字段。
- 界面预览：在 UI 面板展示 Preview Video 入口；存在预览文件时优先用于播放。

## 典型用法

- 将生成的帧批次编码为 MP4 便于分享。
- 编码 RGBA 透明序列：用 WEBM 预览播放，用 MOV（ProRes）导出高质量成品。
- 为成品视频附加音频轨道并输出到最终交付路径。

## 注意与建议

- 节点调用 `ffmpeg` 完成编码与混流。
- 提供 `audio` 时会在临时目录生成中间 WAV，并在编码结束后清理。
