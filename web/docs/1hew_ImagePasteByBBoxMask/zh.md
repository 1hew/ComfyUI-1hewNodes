# Image Paste By BBox Mask - 根据边界遮罩进行变换与粘贴

**节点功能：** `Image Paste By BBox Mask` 将处理后的图像按 `bbox_mask` 所标记的目标区域进行拟合、变换并粘贴到基础图像上。支持可选 `paste_mask`、位置偏移、缩放、旋转与不透明度控制。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `paste_image` | - | IMAGE | - | - | 待粘贴图像；若为 RGBA 将保留其 alpha |
| `base_image` | - | IMAGE | - | - | 粘贴目标的基础图像 |
| `bbox_mask` | - | MASK | - | - | 用于计算拟合与中心的边界框区域遮罩 |
| `paste_mask` | 可选 | MASK | - | - | 可选遮罩，控制粘贴图像的逐像素透明度 |
| `position_x` | - | INT | 0 | -4096–4096 | 相对边界框中心的水平偏移 |
| `position_y` | - | INT | 0 | -4096–4096 | 相对边界框中心的垂直偏移 |
| `scale` | - | FLOAT | 1.0 | 0.1–10.0 | 对边界框内拟合尺寸的缩放系数 |
| `rotation` | - | FLOAT | 0.0 | -3600.0–3600.0 | 旋转角度（度）；正值为顺时针 |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | 全局不透明度；若存在遮罩/alpha 将一并缩放 |
| `apply_paste_mask` | - | BOOLEAN | false | - | 为 true 时先按非空遮罩区域裁剪 `paste_image` 与 `paste_mask` 后再拟合与粘贴 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 变换并粘贴后的基础图像 |
| `mask` | MASK | 在基础图像上标记已粘贴区域的输出遮罩 |

## 功能说明

- 边界框检测：从 `bbox_mask` 提取 bbox 用于拟合与居中。
- 拟合与变换：在保持比例的前提下拟合到 bbox，随后应用 `scale`、`rotation` 与偏移。
- Alpha 与不透明度：处理 RGBA alpha；`opacity` 同步缩放遮罩/alpha。
- 安全放置：在基础图边界内裁剪；若无重叠则返回原图与空遮罩。
- 批量稳健：按批次循环与设备裁剪并堆叠输出。

## 典型用法

- 裁剪回贴：与 `Image Crop With BBox Mask` 联用，将编辑结果回贴到原位置。
- 引导粘贴：提供 `paste_mask` 获得柔和边缘；通过 `position_x`/`position_y` 微调位置。
- 创意变换：结合 `scale` 与 `rotation` 完成合成与修片流程中的变换装配。

## 注意与建议

- 当 `apply_paste_mask=true` 时，粘贴图与遮罩会先按非空遮罩的边界框裁剪后再拟合。
- `rotation` 为正时顺时针旋转；内部实现按 PIL 的旋转方向进行处理。
- 若边界框为空，节点返回原图与全黑输出遮罩。