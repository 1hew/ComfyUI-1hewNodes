# Image Solid - 纯色画布生成

**节点功能**：`Image Solid` 生成纯色图像，可配置 `alpha` 与颜色反转，并输出统一强度的遮罩。支持预设尺寸或自定义宽高、尺寸倍数约束，以及依据参考图批次逐帧对齐的尺寸推断。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `get_image_size` | 可选 | IMAGE | - | - | 参考图像，用于在生成时按帧对齐尺寸 |
| `preset_size` | - | COMBO | `custom` | 预设 | 目标尺寸预设；非 `custom` 时覆盖 `width`/`height` |
| `width` | - | INT | 1024 | 1–8192 | `preset_size=custom` 时的目标宽度 |
| `height` | - | INT | 1024 | 1–8192 | `preset_size=custom` 时的目标高度 |
| `color` | - | STRING | `1.0` | 灰度/HEX/RGB | 基础颜色；支持灰度 `0..1`、`R,G,B` 与 HEX |
| `alpha` | - | FLOAT | 1.0 | 0.0–1.0 | 对颜色通道施加的全局透明度系数 |
| `invert` | - | BOOLEAN | False | - | 颜色通道在应用 `alpha` 前进行反转 |
| `mask_opacity` | - | FLOAT | 1.0 | 0.0–1.0 | 输出遮罩的不透明度强度 |
| `divisible_by` | - | INT | 8 | 1–1024 | 约束输出尺寸为该值的整数倍 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 纯色图像批次 |
| `mask` | MASK | 与图像对齐的遮罩批次，强度为 `mask_opacity` |

## 功能说明

- 尺寸灵活：选择 `preset_size` 或自定义 `width`/`height`；也可按批次从 `get_image_size` 推断每帧尺寸。
- 颜色解析：支持灰度（`0..1`）、HEX（`#RRGGBB`）、`R,G,B` 与命名颜色。
- 透明与反转：应用 `alpha`；`invert=True` 在应用 `alpha` 前反转通道。
- 倍数约束：将尺寸向上取整到可被 `divisible_by` 整除。

## 典型用法

- 构建背景画布并设置遮罩强度。
- 通过 `get_image_size` 与参考批次尺寸一致地生成画布。
- 使用预设与 `divisible_by` 快速准备模型输入尺寸。

## 注意与建议

- 遮罩为全 1，经 `mask_opacity` 缩放后与图像尺寸一致。
- 颜色按末通道布局应用，输出限制在 `[0,1]`。