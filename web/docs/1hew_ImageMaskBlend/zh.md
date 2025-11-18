# Image Mask Blend - 遮罩引导的图像融合

**节点功能：** `Image Mask Blend` 使用灰度遮罩对图像进行复合融合，提供填孔、形态扩展/收缩、高斯羽化、反转、强度缩放与丰富的背景策略控制。输出融合后的 `image` 与处理后的 `mask`，并在批量场景中保持稳健一致。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次 |
| `mask` | - | MASK | - | - | 输入遮罩批次；必要时自动调整到与图像匹配的尺寸 |
| `fill_hole` | - | BOOLEAN | true | - | 填补遮罩孔洞，保障连续形状 |
| `invert` | - | BOOLEAN | false | - | 在形态/羽化完成后反转选择区域 |
| `feather` | - | INT | 0 | 0–50 | 对遮罩进行高斯羽化，平滑边缘 |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | 缩放遮罩强度；0 禁用，1 保持原强度 |
| `expansion` | - | INT | 0 | -100–100 | 正值膨胀、负值腐蚀（像素） |
| `background_color` | - | STRING | 1.0 | 灰度/HEX/RGB/颜色名/`edge`/`average`/`mk`/`mask` | 非遮罩区域的底色来源 |
| `background_opacity` | - | FLOAT | 1.0 | 0.0–1.0 | 非遮罩区域中底色与原图的混合强度 |
| `output_mask_invert` | - | BOOLEAN | false | - | 仅在输出端反转遮罩（不影响融合） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 最终融合：`final = image*mask + mixed_bg*(1-mask)` |
| `mask` | MASK | 处理后的遮罩（`mask_gray * opacity`），可在输出端反转 |

## 功能说明

- 稳健批量：图像与遮罩数量不同时按最大批次循环对齐；必要时以 Lanczos 匹配尺寸。
- 形态操作：支持填孔、膨胀（`expansion>0`）、腐蚀（`expansion<0`）与高斯羽化，提升选区质量。
- 反转与强度：在形态与羽化后执行反转；通过 `opacity` 对选择强度进行统一缩放。
- 背景策略：可从灰度/HEX/RGB、常用颜色名、整图 `average`、图像 `edge` 边缘色或遮罩区域平均色（`mk`/`mask`）解析底色。
- 复合公式：`mixed_bg = (1-background_opacity)*image + background_opacity*background`，确保非遮罩区域的可控融合。

## 典型用法

- 清洁选区：开启 `fill_hole`，配合 `expansion` 与 `feather` 获得平滑边缘与完整形状。
- 柔和过渡：适度 `feather` 与 `opacity<1` 形成自然的边缘衔接。
- 色彩协调：`background_color=average` 或 `mk` 以本图或选区平均色作为底色，提升和谐度。
- 输出控制：下游需要反向遮罩时启用 `output_mask_invert`。

## 注意与建议

- 遮罩尺寸与图像一致时效果最佳；节点会在内部自动调整尺寸。
- 负 `expansion` 可收缩选区；配合小半径 `feather` 可避免锯齿。
- 支持常用颜色名称与简写；不合法输入将回退到白色。