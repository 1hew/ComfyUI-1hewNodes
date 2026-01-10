# Text Encode QwenImageEdit - Qwen 图生编辑提示词编码器

**节点功能：** `Text Encode QwenImageEdit` 将多张图像占位符与文本指令组合为 QwenImageEdit 兼容的输入，并对视觉输入执行尺寸适配，输出 `CONDITIONING`。当连接 `VAE` 时，节点同时写入 `reference_latents` 用于参考图引导。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `clip` | - | CLIP | - | - | QwenImageEdit 兼容的 CLIP，用于 tokenize 与 encode。 |
| `vae` | 可选 | VAE | - | - | 连接后为每张输入图生成 `reference_latents`。 |
| `image_1` | 可选 | IMAGE | - | - | 第一张输入图像；前端支持动态扩展 `image_2..image_10`。 |
| `prompt` | - | STRING(多行) | `""` | - | 用户文本指令，将追加在图像占位符之后。 |
| `reference_skip_prep` | - | COMBO | `first` | `none` / `first` / `all` | 参考潜变量策略：在满足 8 对齐时保留原始尺寸 VAE 编码的范围。 |
| `reference_sq_area` | - | INT | 1024 | 64-8192 | 参考潜变量归一目标面积的边长参数，目标面积为 `reference_sq_area²`。 |
| `vision_embed` | - | COMBO | `stretch` | `crop` / `pad` / `stretch` / `area` | 视觉输入（tokenize 的 images）尺寸适配策略。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `conditioning` | CONDITIONING | 文本/图像条件；当连接 `vae` 时，`reference_latents` 将追加写入 conditioning。 |

## 功能说明

- 多图像提示词：以 `Picture 1..N` 的顺序生成 Qwen 视觉占位符并送入 `clip.tokenize(images=...)`。
- 视觉尺寸策略：提供 `crop`、`pad`、`stretch`、`area`，用于稳定控制视觉编码输入形态。
- 参考潜变量：为每张输入图生成 VAE 潜变量，并作为 `reference_latents` 注入 conditioning。
- 8 对齐保留：依据 `reference_skip_prep` 在满足宽高 8 对齐时采用原图尺寸 VAE 编码。

## `vision_embed` 模式说明

- `area`：等比缩放到目标面积约为 `384×384`，输出宽高由 `sqrt((384²)/(W×H))` 推导。
- `crop`：等比缩放覆盖 `384×384`，随后以中心裁剪获得 `384×384`。
- `pad`：等比缩放以适配进 `384×384`，随后以中心填充补齐到 `384×384`，填充像素值为 0。
- `stretch`：直接缩放到 `384×384`，允许纵横比变化。

## `reference_latents` 规则

- 生成范围：
  - `none`：所有图像统一执行面积归一到 `reference_sq_area²` 并做 8 对齐，再执行 VAE 编码。
  - `first`：第一张图在宽高满足 8 对齐时直接按原尺寸 VAE 编码；其余图像按 `none` 行为处理。
  - `all`：每张图在宽高满足 8 对齐时直接按原尺寸 VAE 编码；其余图像按 `none` 行为处理。
- 归一计算（面积 + 8 对齐）：
  - `scale = sqrt((reference_sq_area²) / (W×H))`
  - `W' = round((W×scale)/8)×8`，`H' = round((H×scale)/8)×8`
  - VAE 对缩放后的 RGB 图像执行编码（取 `:3` 通道）。

## 典型用法

- 文本条件：连接 `clip`，设置 `prompt`，`vae` 留空。
- 图生编辑条件：连接 `clip`，连接一张或多张 `image_*`，选择 `vision_embed`，在 `prompt` 中给出编辑指令。
- 参考引导：连接 `vae`，通过 `reference_skip_prep` 与 `reference_sq_area` 控制参考潜变量行为。

## 注意与建议

- 图像端口按数字顺序处理：`image_1`、`image_2`、`image_3`……
- 在同一工作流中统一使用一种 `vision_embed` 策略可获得更一致的视觉输入形态。
