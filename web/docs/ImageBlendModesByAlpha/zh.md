# Image Blend Modes By Alpha - 图层混合模式（Alpha）

**节点功能：** `Image Blend Modes By Alpha`节点提供全面的图层混合功能，支持基础图层输入、混合模式控制和不透明度调整，可选择性应用遮罩。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `base_image` | 必选 | IMAGE | - | - | 基础图层图像 |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图层图像 |
| `blend_mode` | - | COMBO[STRING] | normal | normal, dissolve, darken, multiply, color burn, linear burn, add, lighten, screen, color dodge, linear dodge, overlay, soft light, hard light, linear light, vivid light, pin light, hard mix, difference, exclusion, subtract, divide, hue, saturation, color, luminosity | 混合模式选择 |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | 叠加图层的不透明度 |
| `overlay_mask` | 可选 | MASK | - | - | 用于选择性混合的可选遮罩 |
| `invert_mask` | - | BOOLEAN | False | True/False | 是否反转叠加遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 混合结果图像 |

## 功能说明

### 混合模式分类
- **正常模式**：normal（正常）、dissolve（溶解）
- **变暗模式**：darken（变暗）、multiply（正片叠底）、color burn（颜色加深）、linear burn（线性加深）
- **变亮模式**：add（相加）、lighten（变亮）、screen（滤色）、color dodge（颜色减淡）、linear dodge（线性减淡）
- **对比模式**：overlay（叠加）、soft light（柔光）、hard light（强光）、linear light（线性光）、vivid light（亮光）、pin light（点光）、hard mix（实色混合）
- **比较模式**：difference（差值）、exclusion（排除）、subtract（减去）、divide（划分）
- **颜色模式**：hue（色相）、saturation（饱和度）、color（颜色）、luminosity（明度）

### 高级功能
- **RGBA支持**：自动将RGBA图像转换为RGB
- **批量处理**：处理不同批次大小的多个图像
- **尺寸适配**：自动调整叠加层尺寸以匹配基础层
- **遮罩集成**：可选遮罩用于选择性混合区域
- **质量处理**：高质量混合算法