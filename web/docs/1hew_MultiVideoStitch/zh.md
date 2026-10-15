# Multi Video Stitch - 多视频首尾拼接

**节点功能：** Multi Video Stitch 将多段视频首尾拼接为一段视频。每段会先用 crop、pad 或 stretch 统一到同一个目标尺寸，因此不同分辨率、不同宽高比的视频也能直接拼接。目标尺寸优先取可选的 get_video_size 参考视频，未连接时取第一段视频的尺寸。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| fps | - | STRING | video_1 | video_N / vN / 数字 | 输出帧率。默认 video_1 沿用第一段视频帧率；video_2 / v2 / v_2 取对应视频帧率；也可直接填 8、29.5 等数字 |
| fit | - | COMBO | pad | crop / pad / stretch | 尺寸适配模式 |
| pad_color | - | STRING | 1.0 | 颜色名/HEX/RGB/edge/average/extend/mirror | pad 模式的补边颜色或策略 |
| video_1 | 可选 | VIDEO | - | - | 第一段视频；后续片段通过动态端口 video_2...video_N 接入 |
| get_video_size | 可选 | VIDEO | - | - | 尺寸参考视频；以它的宽高作为目标尺寸，始终固定在所有 video 端口下方 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| video | VIDEO | 按 video_X 顺序首尾拼接后的单段视频 |

## 功能说明

- 动态片段：按数字后缀收集 video_* 输入，保持 video_1、video_2 … 的顺序。
- 目标尺寸：连接 get_video_size 时以它为准，否则使用第一段有效视频的尺寸。
- 适配模式：stretch 直接拉伸；crop 缩放覆盖后居中裁切；pad 缩放适配后按 pad_color 居中补边。
- 补边策略：支持显式颜色以及 extend、mirror、edge、average。
- 帧率：默认 video_1 沿用第一段帧率；video_N / vN / v_N 取对应片段的帧率；填写数字时所有片段都会重采样到该帧率，并保持各自时长。
- Alpha：任一片段带 alpha 时全部提升为 RGBA，并以 WebM/VP9 输出，由内置编码器在支持时携带 alpha；否则输出 MP4/H.264。
- 音频：有音轨的片段会重采样后依次拼接，缺失的片段补静音以保持时间轴对齐；全部无音轨时不输出音轨。

## 典型用法

- 把不同分辨率或宽高比的片段拼成一段连续视频。
- 在保存或送入后续视频节点前统一一组混合片段。

## 注意与建议

- get_video_size 只提供目标尺寸，不会被拼接进输出。
- fps 引用某一段（如 video_1）且各片段帧率不一致时，每段保持自身时长。
- 结果会编码到临时 MP4/WebM 文件并以视频对象返回。
