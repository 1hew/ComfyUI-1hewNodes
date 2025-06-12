# Image Blend Modes By CSS - 图像混合模式 CSS

**节点功能：** `Image Blend Modes By CSS`节点基于Pilgram库实现CSS标准的图像混合模式，提供与Web CSS混合模式一致的图像合成效果。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图像（上层） |
| `base_image` | 必选 | IMAGE | - | - | 基础图像（下层） |
| `blend_mode` | - | COMBO[STRING] | normal | 16种CSS混合模式 | CSS混合模式选择 |
| `blend_percentage` | - | FLOAT | 1.0 | 0.0-1.0 | 混合强度百分比 |
| `overlay_mask` | 可选 | MASK | - | - | 叠加区域的遮罩 |
| `invert_mask` | 可选 | BOOLEAN | False | True/False | 是否反转遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | CSS混合后的图像 |

## 功能说明

### CSS混合模式类别
#### 基础模式
- **normal**：正常叠加
- **multiply**：正片叠底
- **screen**：滤色模式
- **overlay**：叠加模式

#### 变暗变亮模式
- **darken**：变暗，取较暗值
- **lighten**：变亮，取较亮值
- **color_dodge**：颜色减淡
- **color_burn**：颜色加深

#### 对比模式
- **hard_light**：强光效果
- **soft_light**：柔光效果

#### 差值模式
- **difference**：差值模式
- **exclusion**：排除模式

#### 颜色模式
- **hue**：色相模式
- **saturation**：饱和度模式
- **color**：颜色模式
- **luminosity**：明度模式