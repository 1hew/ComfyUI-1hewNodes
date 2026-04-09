# Image Alpha Split - 提取 alpha 并铺到底色

**节点功能：** `Image Alpha Split` 将输入图像拆出 alpha 作为 `mask` 输出，同时把透明区域按 `background_color` 铺到底色上，输出处理后的 RGB 图像。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像，支持单图或批量；RGBA 会提取 alpha 并与底色合成 |
| `background_color` | - | STRING | `1.0` | 与 `Image Solid` 的 `color` 相同：颜色名 / `#RRGGBB` / `r,g,b` / `0~1` 灰度 / 单字母颜色简写 | 透明区域要铺上的底色 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 铺到底色后的 RGB 图像 |
| `mask` | MASK | 从输入图像提取出的 alpha 遮罩；无 alpha 输入时为全白 mask |

## 功能说明

- RGBA 输入：输出 `mask` 为原始 alpha，同时使用 `background_color` 对透明区域做标准合成。
- RGB 输入：`image` 原样输出，`mask` 返回全白。
- 批量支持：输入是 image batch 时，会逐张处理并同步输出 image 与 mask。
- 颜色输入兼容：与 `Image Solid` 的 `color` 输入规则保持一致。

## 典型用法

- 把透明 PNG 同时拆出 alpha mask，并生成可直接预览或继续处理的底色版图像。
- 给不支持 alpha 的下游节点准备白底图，同时保留透明区域作为 mask。
- 在批处理流程里统一输出“铺底图 + alpha mask”两路结果。

## 注意与建议

- 本节点的 `image` 输出固定为 RGB，不保留 alpha 通道。
- 如果输入本身没有 alpha，`mask` 会是全白，`background_color` 不会影响图像结果。
