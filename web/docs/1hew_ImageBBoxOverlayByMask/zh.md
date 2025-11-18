# Image BBox Overlay by Mask - 基于遮罩绘制边框

**节点功能**：`Image BBox Overlay by Mask` 根据输入遮罩的连通区域绘制边框，或对整张遮罩生成单一合并边框。支持颜色、描边宽度、区域填充与边界外扩。

## Inputs | 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次（`B×H×W×3`） |
| `mask` | - | MASK | - | - | 遮罩批次（`B×H×W`）；自动与图像对齐并广播到批量长度 |
| `bbox_color` | - | COMBO | `green` | 选项 | `red`/`green`/`blue`/`yellow`/`cyan`/`magenta`/`white`/`black` |
| `stroke_width` | - | INT | 4 | 1–100 | 轮廓模式下的描边宽度 |
| `fill` | - | BOOLEAN | True | - | `True` 填充边框区域；`False` 仅绘制轮廓 |
| `padding` | - | INT | 0 | 0–1000 | 在四周按像素外扩边框 |
| `output_mode` | - | COMBO | `separate` | `separate`/`merge` | `separate` 连通组件分别绘制；`merge` 整体合并为单一边框 |

## Outputs | 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 绘制边框后的图像批次 |

## 功能说明

- 批量对齐：将 `mask` 广播到与图像一致的批量长度，并按 LANCZOS 对齐到图像尺寸。
- 模式选择：`separate` 使用 `regionprops` 获取连通组件；`merge` 对所有正值像素取全局最小/最大坐标。
- 边界外扩：按 `padding` 将边框在四个方向外扩，并限制在图像范围内。
- 异步处理：逐样本在工作线程执行，保持交互流畅。

## 典型用法

- 检测可视化：设置 `output_mode=separate`，为每个连通区域绘制边框以观察实例分布。
- 单一区域：设置 `output_mode=merge` 并调整 `padding`，获取整体兴趣区域框。
- 展示风格：切换 `fill` 以填充矩形，或使用 `stroke_width` 配置轮廓粗细。

## 注意与建议

- 组件与合并边框计算基于 8 位表示上的阈值 `>128`。
- 颜色选项映射到固定 RGB 值；未匹配时默认使用 `green`。