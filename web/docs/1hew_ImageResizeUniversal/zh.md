# Image Resize Universal - 图像重置尺寸通用版

**节点功能：** `Image Resize Universal` 是一个功能强大的图像重置尺寸节点，支持多种纵横比、缩放模式和适应方式，可以智能地调整图像大小以满足不同需求。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 可选 | IMAGE | - | - | 要缩放的输入图像 |
| `mask` | 可选 | MASK | - | - | 要缩放的输入遮罩 |
| `get_image_size` | 可选 | IMAGE | - | - | 获取目标尺寸的参考图像，如果提供则使用参考图像的尺寸 |
| `preset_ratio` | - | COMBO[STRING] | origin | origin, custom, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | 预设纵横比选择，origin保持原比例，custom使用自定义比例 |
| `proportional_width` | - | INT | 1 | 1-1e8 | 自定义比例宽度值，用于custom模式 |
| `proportional_height` | - | INT | 1 | 1-1e8 | 自定义比例高度值，用于custom模式 |
| `method` | - | COMBO[STRING] | lanczos | nearest, bilinear, lanczos, bicubic, hamming, box | 图像缩放算法选择 |
| `scale_to_side` | - | COMBO[STRING] | None | None, longest, shortest, width, height, mega_pixels_k | 按边缩放模式，决定如何计算目标尺寸 |
| `scale_to_length` | - | INT | 1024 | 4-1e8 | 目标长度值，配合scale_to_side使用 |
| `fit` | - | COMBO[STRING] | crop | stretch, crop, pad | 适应方式：stretch拉伸、crop裁剪、pad填充 |
| `pad_color` | - | STRING | 1.0 | 灰度值/HEX/RGB/edge | 填充颜色，支持多种格式或使用"edge"自动获取边缘颜色 |
| `divisible_by` | - | INT | 8 | 1-1024 | 尺寸整除数，确保输出尺寸能被指定数字整除 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 缩放后的图像 |
| `mask` | MASK | 缩放后的遮罩 |

## 功能说明

### 纵横比模式
- **origin**：保持原始图像的纵横比
- **custom**：使用proportional_width和proportional_height自定义比例
- **预设比例**：1:1、3:2、4:3、16:9、21:9、2:3、3:4、9:16、9:21等常用比例

### 缩放模式
- **None**：保持原始尺寸，仅调整纵横比
- **longest**：按最长边缩放到指定长度
- **shortest**：按最短边缩放到指定长度
- **width**：按宽度缩放到指定长度
- **height**：按高度缩放到指定长度
- **mega_pixels_k**：按像素总数缩放（以千像素为单位）

### 适应方式
- **stretch**：直接拉伸到目标尺寸，可能改变图像比例
- **crop**：保持比例缩放后裁剪多余部分，居中裁剪
- **pad**：保持比例缩放后用指定颜色填充空白区域

### 缩放算法
- **nearest**：最近邻插值，速度快但质量较低
- **bilinear**：双线性插值，平衡速度和质量
- **lanczos**：Lanczos插值，高质量缩放（默认推荐）
- **bicubic**：双三次插值，高质量但速度较慢
- **hamming**：Hamming窗口插值
- **box**：盒式滤波器

### 填充颜色格式
- **灰度值**：如"0.5"表示50%灰度
- **HEX格式**：如"#FF0000"表示红色
- **RGB格式**：如"255,0,0"或"1.0,0.0,0.0"
- **edge**：自动使用图像边缘的平均颜色进行填充
