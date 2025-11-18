# Image Blend Mode by Alpha - 图层透明度混合

**节点功能：** `Image Blend Mode by Alpha` 节点将叠加图层以多种专业混合模式与基础图层融合，并通过整体 `opacity` 控制强度。支持可选的逐像素 `overlay_mask`、RGBA→RGB 规范化、尺寸对齐与稳健的批量处理。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | - | IMAGE | - | - | 叠加图层批次，覆盖在 `base_image` 之上 |
| `base_image` | - | IMAGE | - | - | 基础图层批次 |
| `overlay_mask` | 可选 | MASK | - | - | 可选遮罩；控制在何处用融合结果替换基础图 |
| `blend_mode` | - | COMBO | `normal` | `normal` / `dissolve` / `darken` / `multiply` / `color_burn` / `linear_burn` / `add` / `lighten` / `screen` / `color_dodge` / `linear_dodge` / `overlay` / `soft_light` / `hard_light` / `linear_light` / `vivid_light` / `pin_light` / `hard_mix` / `difference` / `exclusion` / `subtract` / `divide` / `hue` / `saturation` / `color` / `luminosity` | 混合算法选择 |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | 全局强度；0 仅基础图，1 完全应用叠加混合 |
| `invert_mask` | - | BOOLEAN | false | - | 在应用前反转 `overlay_mask` |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 融合后的图像批次 |

## 功能说明

- 专业混合模式：包含正常/溶解、变暗/变亮、正片叠底/滤色、加深/减淡（含线性）、相加/相减/相除、叠加/柔光/强光/线性光/亮光/点光/实色混合、差值/排除，以及基于 HSL 的色相/饱和度/颜色/明度。
- RGBA 规范化：将含透明通道的输入以白色背景展平为 RGB，保证混合一致性。
- 尺寸对齐：自动将叠加图层调整到与基础图层相同尺寸。
- 遮罩混合：`overlay_mask` 控制替换区域；`invert_mask` 反转选择；与 `opacity` 联合实现精细控制。
- 批量稳健性：批次数不一致时循环扩展较小批次；遮罩批次按图像批次轮询使用。
- 设备安全：在张量运算中保持设备一致。

## 典型用法

- 明暗塑形：`multiply` 用于压暗，`screen` 用于提亮。
- 对比增强：`overlay` / `hard_light` 强化细节与对比度。
- 色彩协调：`hue` / `saturation` / `color` / `luminosity` 调整色彩属性但保留结构。
- 区域限定：提供 `overlay_mask` 仅在选区内应用；必要时启用 `invert_mask`。

## 注意与建议

- `opacity` 为整体强度缩放；结合遮罩可获得更精准的空间控制。
- `divide` 内部做零值保护并裁剪结果到合法范围。
- `dissolve` 含随机性；提高 `opacity` 增加叠加像素比例。
- RGBA 输入以白色展平；如需其他背景假设，可在前处理阶段自定义。