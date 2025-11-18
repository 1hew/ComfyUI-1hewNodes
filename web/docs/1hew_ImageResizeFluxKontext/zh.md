# Image Resize FluxKontext - 预设模型分辨率

**节点功能**：`Image Resize FluxKontext` 将图像/遮罩缩放到 FluxKontext 预设分辨率。支持 `auto` 最近纵横比匹配、`crop`/`pad`/`stretch` 三种适配模式，以及多种 `pad_color` 填充策略。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `preset_size` | - | COMBO | `auto` | 预设 | 目标分辨率；`auto` 选择与输入最接近的纵横比预设 |
| `fit` | - | COMBO | `crop` | `crop`/`pad`/`stretch` | 目标尺寸的适配策略 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/`edge`/`average`/`extend`/`mirror` | 填充背景策略 |
| `image` | 可选 | IMAGE | - | - | 输入图像批次 |
| `mask` | 可选 | MASK | - | - | 输入遮罩批次；严格与图像对齐或单独使用 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 按预设分辨率缩放后的图像批次 |
| `mask` | MASK | 与输出几何对齐的遮罩 |

## 功能说明

- `auto` 预设：根据输入图像/遮罩的纵横比选择最接近的预设。
- 适配模式：
  - `crop`：居中裁剪到目标纵横比后缩放；遮罩标记原始裁剪窗口。
  - `pad`：保留纵横比居中填充，输出遮罩标识原始内容区域。
  - `stretch`：图像与遮罩直接缩放。
- 遮罩处理：标准化为 `B×H×W`，使用最近邻重采样并保持 `[0,1]` 范围。
- 颜色解析：支持灰度、HEX、`R,G,B` 与命名策略（`edge`/`average`/`extend`/`mirror`）。

## 典型用法

- 模型对齐：将 `preset_size` 选择为 FluxKontext 预设，或使用 `auto`。
- 保留内容：设置 `fit=pad` 并选用 `edge`/`average` 以获得自然边界。
- 严格内容区域：设置 `fit=crop` 清晰地去除多余区域。

## 注意与建议

- 仅提供 `mask` 时，节点将依据 `pad_color` 生成统一颜色图像并对遮罩进行缩放。
- 输出采用末通道布局并限制在 `[0,1]`。