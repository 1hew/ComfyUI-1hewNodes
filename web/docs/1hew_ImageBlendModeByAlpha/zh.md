# Image Blend Mode by Alpha - 图层叠加模式（Alpha）

**节点功能：** `Image Blend Mode by Alpha` 以图层叠加模式将叠加图（overlay）融合到基础图（base），可调不透明度，支持遮罩，自动处理 RGBA→RGB、尺寸对齐与批次循环。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图像批次 |
| `base_image` | 必选 | IMAGE | - | - | 基础图像批次 |
| `blend_mode` | 必选 | COMBO[STRING] | normal | 模式列表 | 混合模式：`normal`, `dissolve`, `darken`, `multiply`, `color_burn`, `linear_burn`, `add`, `lighten`, `screen`, `color_dodge`, `linear_dodge`, `overlay`, `soft_light`, `hard_light`, `linear_light`, `vivid_light`, `pin_light`, `hard_mix`, `difference`, `exclusion`, `subtract`, `divide`, `hue`, `saturation`, `color`, `luminosity` |
| `opacity` | 必选 | FLOAT | 1.0 | 0.0–1.0 | 混合不透明度强度 |
| `overlay_mask` | 可选 | MASK | - | - | 遮罩（应用于融合结果） |
| `invert_mask` | 可选 | BOOLEAN | False | True/False | 在应用前反转遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 按所选模式与不透明度融合后的图像 |
