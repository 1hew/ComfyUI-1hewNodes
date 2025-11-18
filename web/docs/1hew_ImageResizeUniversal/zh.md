# Image Resize Universal - 图像通用缩放器

**节点功能：** `Image Resize Universal` 节点用于在多种比例和适应模式下对图像进行统一缩放，同时提供完备的遮罩输出策略。支持纵横比预设、目标边控制、采样方法选择、尺寸倍数约束，以及多种填充背景策略。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 可选 | IMAGE | - | - | 输入图像批次，未提供时可依据 `preset_ratio`/`get_image_size` 推导输出尺寸 |
| `mask` | 可选 | MASK | - | - | 输入遮罩批次，与图像尺寸一致时随图像同步缩放；缺省时自动生成默认遮罩 |
| `get_image_size` | 可选 | IMAGE | - | - | 仅用于获取尺寸的参考图像（取第一张），用于目标尺寸推导 |
| `preset_ratio` | - | COMBO | `origin` | `origin` / `custom` / `1:1` / `3:2` / `4:3` / `16:9` / `21:9` / `2:3` / `3:4` / `9:16` / `9:21` | 纵横比来源；`origin` 使用输入尺寸，`custom` 使用下方比例参数 |
| `proportional_width` | - | INT | 1 | 1-8192 | `custom` 模式下的比例宽 |
| `proportional_height` | - | INT | 1 | 1-8192 | `custom` 模式下的比例高 |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | 缩放采样方法 |
| `scale_to_side` | - | COMBO | `None` | `None` / `longest` / `shortest` / `width` / `height` / `length_to_sq_area` | 目标边控制；例如按最长边或指定宽/高设定目标长度 |
| `scale_to_length` | - | INT | 1024 | 1-8192 | 当选择 `scale_to_side` 为上述模式时的目标长度 |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | 适应模式：裁剪、填充或拉伸 |
| `pad_color` | - | STRING | 1.0 | 灰度/HEX/RGB/`edge`/`average`/`extend`/`mirror` | 填充背景策略；详见下文“填充策略” |
| `divisible_by` | - | INT | 8 | 1-1024 | 输出尺寸按该数值取整倍（常用于模型输入尺寸要求） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 缩放后的图像批次；在无输入图像时可生成背景色图 |
| `mask` | MASK | 与输出图像尺寸相匹配的遮罩批次；始终按规则生成 |

## 功能说明

- 纵横比推导：支持原图、标准比例与自定义比例三种来源，联合目标边控制计算目标宽高。
- 尺寸约束：`divisible_by` 保障输出宽高为指定数值的整数倍。
- 采样方法：提供 `nearest`、`bilinear`、`lanczos`、`bicubic`、`hamming`、`box` 等采样器以兼顾质量与性能。
- 适应模式：
  - `crop`：保持比例缩放后居中裁剪到目标尺寸。
  - `pad`：保持比例缩放后居中填充到目标尺寸，严格区分原图与填充区域。
  - `stretch`：直接拉伸到目标尺寸。
- 无图像输入：依据 `preset_ratio`/`scale_to_side` 或 `get_image_size` 推导目标尺寸并生成背景色图像。
- 遮罩生成：在任意输入组合下输出有效遮罩；当存在输入遮罩时随图像同步缩放。

## 遮罩规则

- `pad` 模式：遮罩以目标尺寸输出，原图区域为白色（255），填充区域为黑色（0）。
- `crop` 模式：遮罩以原图尺寸表达裁剪范围白色区域，随后转换为输出遮罩。
- `stretch` 模式：遮罩与输出尺寸一致，整体为白色（255）。
- 缺省遮罩：在未提供遮罩时，按上述规则为每张输出图像生成对应遮罩。

## 填充策略（`pad_color`）

- 灰度值：如 `0.5` 表示 50% 灰度，自动转换为 RGB。
- HEX：如 `#FF0000` 或 `FF0000`。
- RGB：如 `255,0,0` 或 `0.5,0.2,0.8`（0-1 范围自动转换）。
- `edge`：按图像边缘平均色填充（上下或左右）。
- `average`：按整图平均色填充。
- `extend`：复制边缘像素进行填充（`replicate`）。
- `mirror`：镜像边缘像素进行填充（`reflect`，含分段扩展）。

## 典型用法

- 保持比例并控制最长边：设置 `preset_ratio=origin`、`scale_to_side=longest`、`scale_to_length` 为目标长度。
- 固定输入到模型尺寸：设置 `fit=pad`，并设定 `divisible_by=8/16` 以满足模型对齐要求。
- 批量缩放并输出遮罩：同时连接 `image` 与 `mask`，节点将对两者做一致变换并输出对应批次结果。

## 注意与建议

- 图像与遮罩尺寸一致时可获得最稳定的缩放与遮罩同步效果；仅提供遮罩也可用作尺寸参考。
- 无图像输入场景建议同时提供 `get_image_size` 或明确的比例与目标边，以获得确定的输出尺寸。