# Image Blend Mode by CSS - CSS 混合模式

**节点功能：** `Image Blend Mode by CSS` 节点以接近 CSS 的混合语义在基础图层与叠加图层之间进行融合，支持 `blend_mode`、整体 `blend_percentage` 不透明度与可选的 `overlay_mask` 控制，含 HSL 模式：`hue`、`saturation`、`color`、`luminosity`。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | - | IMAGE | - | - | 叠加图层批次，覆盖在 `base_image` 之上 |
| `base_image` | - | IMAGE | - | - | 基础图层批次 |
| `blend_mode` | - | COMBO | `normal` | `normal` / `multiply` / `screen` / `overlay` / `darken` / `lighten` / `color_dodge` / `color_burn` / `hard_light` / `soft_light` / `difference` / `exclusion` / `hue` / `saturation` / `color` / `luminosity` | 选择 CSS 混合算法 |
| `blend_percentage` | - | FLOAT | 100.0 | 0–100 | 效果强度（百分比），控制融合结果整体不透明度 |
| `overlay_mask` | 可选 | MASK | - | - | 可选遮罩；限定在遮罩区域内替换基础图 |
| `invert_mask` | - | BOOLEAN | false | - | 在应用前反转 `overlay_mask` |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 融合后的图像批次 |

## 功能说明

- CSS 公式：实现 multiply、screen、overlay、darken、lighten、color dodge/burn、hard/soft light、difference、exclusion 等典型 CSS 混合语义。
- HSL 模式：`hue`、`saturation`、`color`、`luminosity` 通过 RGB↔HSL 转换实现。
- 强度控制：`blend_percentage`（0–100）在线性混合基础图与融合结果。
- RGBA 规范化：以白色背景展平 alpha，保证 RGB 混合一致性。
- 尺寸与遮罩：自动将叠加图层与遮罩调整到基础图尺寸，并对遮罩进行通道广播。
- 批量处理：当批次数不一致时按最大批次循环处理并对齐输出。

## 典型用法

- 影调处理：`multiply` 压暗、`screen` 提亮、`overlay` 增强对比。
- 色彩属性迁移：使用 HSL 模式传递色相/饱和度/明度而保持结构不变。
- 区域限定：连接 `overlay_mask` 以仅在选区内混合，必要时开启 `invert_mask`。
- 强度调节：通过 `blend_percentage` 精细控制最终效果强弱。

## 注意与建议

- 当尺寸不一致时先以 Lanczos 重采样叠加图层再进行混合。
- HSL 转换阶段对数值做范围约束，结果以 RGB 返回。
- `color_dodge` / `color_burn` 内部做边界防护，避免无穷与负值。