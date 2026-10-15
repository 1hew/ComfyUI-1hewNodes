# Image Resize Area - 按面积自动缩放

**节点功能：** `Image Resize Area` 读取输入图像面积，推导参考正方形边长 `reference = clamp(√(宽 × 高), min_sq_area, max_sq_area)`，把图像与遮罩**按原比例**缩放到面积约 `reference²` 的尺寸；随后与 `Image Resize Universal` 一致，在**所有分支**把输出宽高向上取整到 `divisible_by` 的倍数，因此最终分辨率始终对齐。超过上限用 `method` 缩小；低于下限用 `method` 放大，或接入 `upscale_model` 时先模型放大。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次（`B×H×W×C`）；以第一帧的 H/W 决定目标尺寸，保留 alpha。 |
| `mask` | 可选 | MASK | - | - | 可选遮罩批次，需与图像尺寸一致；按相同的 `fit` / `pad_color` 规则变换。未连接时自动生成默认遮罩。 |
| `upscale_model` | 可选 | UPSCALE_MODEL | - | - | 可选超分模型（连接 `Load Upscale Model`）。仅在低于 `min_sq_area` 时生效，模型输出会再用 `method` 缩放到目标尺寸。懒求值：不需要时上游加载器完全跳过。 |
| `min_sq_area` | - | INT | `1024` | `1-8192` | 参考正方形边长的下限；`√面积 < min_sq_area` 的图像会被放大。目标面积 = `reference²`。 |
| `max_sq_area` | - | INT | `2048` | `1-8192` | 参考正方形边长的上限；`√面积 > max_sq_area` 的图像会被缩小。 |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | 缩放使用的采样方法。 |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | 适配方式，与 `Image Resize Universal` 一致；只要对齐后的目标尺寸与原图不同就会生效。 |
| `pad_color` | - | STRING | `1.0` | 灰度 / HEX / RGB / `edge` / `average` / `extend` / `mirror` | `fit = pad` 时的填充策略，与 `Image Resize Universal` 一致。 |
| `divisible_by` | - | INT | `32` | `1-1024` | 输出宽高都会向上取整到该倍数，与 `Image Resize Universal` 完全一致。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 缩放后的图像：保持原比例的目标尺寸（面积约 `reference²`、宽高为 `divisible_by` 倍数）；对齐后的目标尺寸与原图一致时原样透传。 |
| `mask` | MASK | 与 `image` 同步的遮罩；未连接 `mask` 时输出生成的默认遮罩。 |
| `resolution` | INT | 夹取并对齐 `divisible_by` 后的参考正方形边长 `N`；目标面积约 `N²`。 |

## 功能说明

- 保持原比例：`reference = clamp(√(宽 × 高), min_sq_area, max_sq_area)` 只归一像素总量，目标尺寸沿用原图纵横比，**不会强制成正方形**。例如 4000×3000 输入、`max_sq_area = 2048` 时输出 2368×1792。
- 倍数对齐：与 `Image Resize Universal` 一致，任何分支的输出宽高都向上取整到 `divisible_by` 的倍数；面积已在区间内的图像也会被对齐。
- 三分支行为：
  - `√面积` 在 `[min_sq_area, max_sq_area]` 内：目标尺寸即原图尺寸，只有 `divisible_by` 可能把它略微放大；
  - 高于 `max_sq_area`：用 `method` 缩小；
  - 低于 `min_sq_area`：用 `method` 放大，或先过 `upscale_model` 再用 `method` 归一到目标尺寸。
- 共用缩放内核：`fit` / `pad_color` / 遮罩变换复用 `Image Resize Universal`，crop/pad/stretch、alpha 处理与默认遮罩行为完全一致。
- 模型放大：等价于官方 `Upscale Image (using Model)`，分块处理并在显存不足时自动降低 tile。

## 遮罩规则

- 接了 `mask`：与图像使用相同的 `fit` 和采样器同步缩放。
- 未接 `mask` 且 `fit = pad`：目标尺寸，原图区域白（255）、填充区域黑（0）。
- 未接 `mask` 且 `fit = crop` / `stretch`：目标尺寸全白（255）。
- 目标尺寸与原图一致时：接了遮罩则原样返回，未接则生成与原图同尺寸的全白遮罩。

## 填充策略（`pad_color`）

与 `Image Resize Universal` 相同：灰度（`0.5`）、HEX（`#FF0000`）、RGB（`255,0,0` 或 `0.5,0.2,0.8`）、`edge`、`average`、`extend`、`mirror`。

## 典型用法

- 有界归一：把 `√面积` 限制在 `min_sq_area`~`max_sq_area`，得到保持原比例且 32 对齐的结果给模型。
- 质量放大：接上 `Load Upscale Model`，只有低于 `min_sq_area` 的图才会真正跑模型。
- 遮罩同步流程：同时输入 image + mask，输出保持同尺寸。

## 注意与建议

- 只有第一帧的空间尺寸决定目标尺寸，整批一起缩放。
- 目标尺寸使用与 `Image Resize Universal` 的 `length_to_sq_area` 相同的 `√ratio × reference` 公式，再把宽高向上取整到 `divisible_by`。
- `min_sq_area > max_sq_area` 时会自动交换上下限。
- 若区间内不存在 `divisible_by` 的倍数，节点不再报错：会退回区间内最接近的整数参考边长，输出宽高仍会对齐。
- 原图尺寸恰好等于对齐后的目标尺寸时，图像 / 遮罩原样透传，不会做无谓重采样。
- `upscale_model` 是懒求值：不需要它的图像会完全跳过上游 `Load Upscale Model`。
