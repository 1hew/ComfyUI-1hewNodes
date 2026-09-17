# Image Minimum Area - 最小面积放大

**节点功能：** `Image Minimum Area` 将图像放大到面积达到最小方形参考面积。`length_to_sq_area` 定义参考方形边长 `N`（目标面积 = `N²`）。`fit` 行为与 `Image Resize Universal` 一致（`crop` / `pad` / `stretch`），`divisible_by` 将最终宽高向上取整到指定倍数，`resize_bool` 报告是否真正发生了放大。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 待放大的输入图像批次 |
| `mask` | 可选 | MASK | - | - | 随图像同步缩放的可选遮罩；未提供时自动生成默认遮罩 |
| `length_to_sq_area` | - | INT | 1024 | 1–65536 | 参考方形边长 `N`；目标面积 = `N²` |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | 缩放采样方法 |
| `divisible_by` | - | INT | 1 | 1–1024 | 将最终宽高向上取整到该值的整数倍 |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | 缩放时的适应模式 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB | `fit=pad` 时的填充背景色 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 放大后的图像批次；面积已达标时直接原样透传 |
| `mask` | MASK | 同步缩放后的输入遮罩，或按输出尺寸生成的默认遮罩 |
| `resize_bool` | BOOLEAN | 是否实际执行了放大，是为 `True`，否则为 `False` |

## 功能说明

- 最小面积目标：按单一缩放因子同时缩放两边，使结果面积达到 `length_to_sq_area²`。
- 通用式适应：`crop` / `pad` / `stretch` 与 `Image Resize Universal` 行为一致。
- 倍数约束：`divisible_by` 可为最终宽高补足少量像素，以满足模型尺寸要求。
- 遮罩同步：输入遮罩随图像一起缩放；未提供时自动生成默认遮罩。
- 透传：输入面积已达标时全部原样返回，`resize_bool` 为 `False`。

## 典型用法

- 为需要一定像素面积的下游模型保证最低分辨率。
- 将小尺寸裁剪图放大到标准面积后再继续处理。

## 注意与建议

- `divisible_by=1` 表示关闭倍数约束。
- 提供的遮罩会以与图像相同的采样器与适应模式进行缩放。
