# Image Blend Mode by CSS - CSS 图层叠加模式

**节点功能：** `Image Blend Mode by CSS` 使用 `pilgram` 库实现 CSS 混合模式，将叠加图与基础图进行融合。支持遮罩应用、混合强度（百分比）、RGBA→RGB 合成以及批次循环。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图像批次 |
| `base_image` | 必选 | IMAGE | - | - | 基础图像批次 |
| `blend_mode` | 必选 | COMBO[STRING] | normal | 模式列表 | CSS 混合模式：`normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, `color_dodge`, `color_burn`, `hard_light`, `soft_light`, `difference`, `exclusion`, `hue`, `saturation`, `color`, `luminosity` |
| `blend_percentage` | 必选 | FLOAT | 1.0 | 0.0–1.0 | CSS 混合强度（应用后比例缩放） |
| `overlay_mask` | 可选 | MASK | - | - | 遮罩用于对融合结果进行空间混合 |
| `invert_mask` | 可选 | BOOLEAN | False | True/False | 遮罩在应用前反转 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 使用所选 CSS 模式与强度融合后的图像 |
