# Image Mask Crop - 遮罩边界裁剪与 Alpha 输出

**节点功能：** `Image Mask Crop` 根据遮罩的边界框进行裁剪或保持原尺寸，并可选地将遮罩作为 alpha 通道输出。返回图像与处理后的遮罩，并在批量场景下自动统一尺寸。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次 |
| `mask` | - | MASK | - | - | 输入遮罩批次 |
| `output_crop` | - | BOOLEAN | true | - | 按遮罩边界框裁剪；为 false 时保持原图尺寸 |
| `output_alpha` | - | BOOLEAN | false | - | 为 true 时输出 RGBA（alpha 来自遮罩）；否则输出 RGB |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 裁剪或原尺寸输出；`output_alpha=true` 时为 RGBA |
| `mask` | MASK | 与图像输出对齐的裁剪或居中填充遮罩 |

## 功能说明

- 边界框裁剪：从遮罩计算 bbox 并同步裁剪图像/遮罩。
- Alpha 通道：`output_alpha=true` 时将遮罩写入图像 alpha。
- 尺寸保持：`output_crop=false` 时保留原尺寸，并在需要时居中对齐并填充遮罩。
- 批量稳健：图像与遮罩数量不一致时循环使用；输出在堆叠前做统一填充。

## 典型用法

- 抠像剪切：设 `output_alpha=true` 生成带 alpha 的 RGBA；后续合成可直接利用透明度。
- 紧凑裁剪：设 `output_crop=true` 聚焦遮罩区域；`output_alpha=false` 保持纯 RGB。
- 全画布输出：设 `output_crop=false` 保持原始画布，同时应用遮罩 alpha。

## 注意与建议

- 当遮罩全黑时，将回退为原尺寸输出；若启用 alpha，则仍会应用 alpha 通道。
- 如输入为 RGBA，除非启用 `output_alpha`，裁剪流程会统一为 RGB。
- 输出在设备上做范围裁剪到 [0,1]。