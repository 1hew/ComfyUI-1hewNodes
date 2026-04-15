# Image Alpha Join - 合并 image 与 alpha

**节点功能：** `Image Alpha Join` 将可选的 `image` 与 `mask` 合并为一个 4 通道 IMAGE 输出，用于生成带 alpha 的 RGBA 图像。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 可选输入图像，支持单图或批量；若含 alpha，仅使用其 RGB 部分 |
| `mask` | - | MASK | - | `0~1` | 可选 alpha 遮罩；与 `image` 同时输入时会作为输出图像的 alpha |
| `invert_mask` | - | BOOLEAN | `true` | `true` / `false` | 是否先反转 `mask` 再写入 alpha；默认开启 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 合并后的 4 通道图像（RGBA） |

## 功能说明

- 两者都不输入：输出空。
- 只输入 `image`：自动补全为全白 alpha，整张图完全显示。
- 只输入 `mask`：输出与 `mask` 同尺寸的完全透明图，仅借用尺寸，不使用 `mask` 数值。
- 同时输入 `image` 和 `mask`：输出 `image` 的 RGB 与 `mask` 的 alpha 合并结果；若 `invert_mask=true`，则会先反转 `mask`。
- 尺寸不一致时：会将 `mask` 以最近邻方式缩放到 `image` 尺寸。
- 批量支持：`image` 与 `mask` 任一侧 batch 更短时，会按顺序循环复用。

## 典型用法

- 把 RGB 图与独立遮罩重新合成为 RGBA 图像。
- 在需要 4 通道 IMAGE 的流程里，给普通图像快速补一个全 alpha。
- 用纯透明占位图维持下游 RGBA 流程的尺寸一致性。

## 注意与建议

- 输出固定为 4 通道 IMAGE。
- `image` 输入若本身已有 alpha，本节点不会保留原 alpha，而是以外部 `mask` 或默认全白 alpha 为准。
- 默认开启 `invert_mask`，更适合“白色表示要抠掉/透明”的遮罩流；若你的 `mask` 本身就是标准 alpha，请关闭它。
