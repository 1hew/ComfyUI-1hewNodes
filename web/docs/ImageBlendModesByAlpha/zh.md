# Image Blend Modes By Alpha - 图像混合模式 alpha

**节点功能：** `Image Blend Modes By Alpha`节点提供多种专业的图像混合模式，支持透明度控制和遮罩应用，实现类似Photoshop的图层混合效果。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图像（上层） |
| `base_image` | 必选 | IMAGE | - | - | 基础图像（下层） |
| `blend_mode` | - | COMBO[STRING] | normal | 多种混合模式 | 混合模式选择 |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | 叠加图像的不透明度 |
| `overlay_mask` | 可选 | MASK | - | - | 叠加区域的遮罩 |
| `invert_mask` | 可选 | BOOLEAN | False | True/False | 是否反转遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 混合后的图像 |

## 功能说明

### 混合模式类别

#### 基础模式
- **normal**：正常叠加
- **dissolve**：溶解效果，随机丢弃像素

#### 变暗模式
- **darken**：变暗，取较暗值
- **multiply**：正片叠底，颜色相乘
- **color burn**：颜色加深
- **linear burn**：线性加深

#### 变亮模式
- **lighten**：变亮，取较亮值
- **screen**：滤色模式
- **color dodge**：颜色减淡
- **linear dodge**：线性减淡
- **add**：相加模式

#### 对比模式
- **overlay**：叠加模式
- **soft light**：柔光效果
- **hard light**：强光效果
- **linear light**：线性光
- **vivid light**：亮光模式
- **pin light**：点光模式
- **hard mix**：实色混合

#### 差值模式
- **difference**：差值模式
- **exclusion**：排除模式
- **subtract**：减去模式
- **divide**：除法模式

#### 颜色模式
- **hue**：色相模式
- **saturation**：饱和度模式
- **color**：颜色模式
- **luminosity**：明度模式