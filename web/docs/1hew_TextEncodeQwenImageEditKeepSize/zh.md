# Text Encode QwenImageEdit Keep Size - 文本编码 QwenImageEdit 保持尺寸

**节点功能：** `Text Encode QwenImageEdit Keep Size` 节点用于为 Qwen/VL 图像编辑生成 Conditioning，将用户文本与一个或多个视觉输入结合。在提供 `VAE` 时，按 `keep_size` 策略保持原图尺寸或依据 `base_size` 进行面积等比缩放并对齐到 8 的倍数，同时将编码后的 `reference_latents` 附加到 Conditioning。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `clip` | 必选 | CLIP | - | - | 文本/视觉编码器（Qwen/CLIP），用于生成 Conditioning |
| `prompt` | 必选 | STRING | "" | 多行文本 | 编辑指令文本，支持多行 |
| `keep_size` | 必选 | COMBO[STRING] | first | none, first, all | 在提供 `VAE` 时的尺寸保持策略 |
| `base_size` | 必选 | COMBO[STRING] | 1024 | 1024, 1536, 2048 | 面积等比缩放的基准边长，并在编码前对齐到 8 的倍数 |
| `vae` | 可选 | VAE | - | - | 用于编码并附加 `reference_latents` |
| `image_1` | 可选 | IMAGE | - | - | 输入图像，支持扩展为 `image_2`、`image_3` 等动态端口 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `conditioning` | CONDITIONING | 包含文本与视觉编码的 Conditioning，可附带 `reference_latents` |
