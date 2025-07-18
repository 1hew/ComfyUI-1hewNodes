# Image Edge Crop Pad - 图像边缘裁剪填充

**节点功能：** `Image Edge Crop Pad`节点用于对图像进行边缘裁剪或填充操作，支持负数值向内裁剪和正数值向外填充，支持多种颜色格式和边缘填充模式，常用于图像尺寸调整和边缘处理。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 要处理的输入图像 |
| `left_amount` | - | FLOAT | 0 | -8192~8192 | 左侧裁剪/填充量，负数为裁剪，正数为填充 |
| `right_amount` | - | FLOAT | 0 | -8192~8192 | 右侧裁剪/填充量，负数为裁剪，正数为填充 |
| `top_amount` | - | FLOAT | 0 | -8192~8192 | 顶部裁剪/填充量，负数为裁剪，正数为填充 |
| `bottom_amount` | - | FLOAT | 0 | -8192~8192 | 底部裁剪/填充量，负数为裁剪，正数为填充 |
| `uniform_amount` | - | FLOAT | 0 | -8192~8192 | 统一裁剪/填充量，当不为0时覆盖其他方向的设置 |
| `pad_color` | - | STRING | 0.0 | 多种格式 | 填充颜色，支持灰度值、HEX、RGB、颜色名称和特殊值 |
| `divisible_by` | - | INT | 8 | 1-1024 | 确保输出尺寸能被指定数值整除，常用于AI模型的尺寸要求 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 处理后的图像 |
| `mask` | MASK | 操作区域遮罩，裁剪或填充区域为白色，原图区域为黑色 |

## 功能说明

### 数值模式
- **百分比模式**：当绝对值小于1时（如0.1、-0.2），按图像尺寸的百分比计算
- **像素模式**：当绝对值大于等于1时（如50、-100），直接使用像素值
- **负数值**：向内裁剪，减少图像尺寸
- **正数值**：向外填充，增加图像尺寸

### 统一模式
- **uniform_amount优先**：当uniform_amount不为0时，会覆盖其他四个方向的设置
- **智能分配**：负数时向内裁剪各边，正数时向外填充各边
- **百分比处理**：统一模式下的百分比会自动分配到各边（除以2）

### 填充颜色支持
- **灰度值**：如"0.5"表示50%灰度，"1.0"表示白色
- **HEX格式**：如"#FF0000"或"FF0000"表示红色
- **RGB格式**：如"255,0,0"或"1.0,0.0,0.0"表示红色
- **颜色名称**：如"red"、"blue"、"white"等标准颜色名称
- **边缘颜色**：使用"edge"、"e"或"ed"自动计算各边缘的平均颜色进行填充
- **平均颜色**：使用"average"、"avg"或"a"计算整个图像的平均颜色