# Image Crop With BBox Mask - 遮罩裁剪并返回边界框遮罩

**节点功能：** `Image Crop With BBox Mask` 以遮罩的边界框为中心进行裁剪，支持可配置的纵横比与边长目标控制。节点同时返回裁剪后的图像、原图尺寸的 `bbox_mask`（在裁剪矩形区域内为白色）与 `cropped_mask`（与裁剪输出对齐）。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次 |
| `mask` | - | MASK | - | - | 引导裁剪中心与边界框；必填 |
| `preset_ratio` | - | COMBO | `mask` | `mask` / `image` / `auto` / `1:1` / `3:2` / `4:3` / `16:9` / `21:9` / `2:3` / `3:4` / `9:16` / `9:21` | 纵横比来源 |
| `scale_strength` | - | FLOAT | 0.0 | 0.0–1.0 | 在可行尺寸的最小/最大范围中选择具体目标；值越大裁剪越大 |
| `crop_to_side` | - | COMBO | `None` | `None` / `longest` / `shortest` / `width` / `height` | 边长控制模式 |
| `crop_to_length` | - | INT | 1024 | 8–8192 | 当使用 `crop_to_side` 时的目标边长 |
| `divisible_by` | - | INT | 8 | 1–1024 | 目标宽高需为该数值的整数倍 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `cropped_image` | IMAGE | 裁剪后的图像；若批次内尺寸不一致将统一填充对齐 |
| `bbox_mask` | MASK | 原图尺寸遮罩；裁剪矩形区域为白色（1.0） |
| `cropped_mask` | MASK | 与裁剪输出一致的遮罩 |

## 功能说明

- 比例方向：据 `preset_ratio` 判断横/竖/方形方向。
- 灵活尺寸：生成满足比例与 `divisible_by` 的候选尺寸。
- 边长控制：按 `width`/`height` 或 `longest`/`shortest` 结合 `crop_to_length` 设定裁剪目标。
- 边界框遮罩：输出 `bbox_mask` 在裁剪区域为 1，其余为 0。
- 批量稳健：扩展较小批次并在堆叠前对图像与遮罩做统一填充。

## 典型用法

- 固定比例：设 `preset_ratio=4:3` 与 `divisible_by=8/16`，便于与模型输入对齐。
- 控制长边：`crop_to_side=longest` 搭配 `crop_to_length`，在多样输入上保持统一边长。
- 回贴流程：使用 `bbox_mask` 配合 `Image Paste By BBox Mask` 将处理后的裁剪结果回贴到原位置。

## 注意与建议

- 遮罩为空或无效时，回退为原图与全白 `bbox_mask` 输出。
- 裁剪输出在设备上做范围约束，并将 RGBA 统一为 RGB。
- 批次尺寸差异会通过填充统一后再进行堆叠输出。