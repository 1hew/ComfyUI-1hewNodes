# Image Solid FluxKontext - 图像纯色 FluxKontext

**节点功能：** `Image Solid FluxKontext` 节点基于 Flux Kontext 尺寸预设生成纯色图像，支持为 Flux 模型工作流优化的多种宽高比。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `preset_size` | - | COMBO[STRING] | 1328×1328 [1:1.00] (1:1) | Flux Kontext 预设选项 | 为 Flux 模型优化的预设尺寸选择，包含从 1:2.33 到 2.33:1 的比例 |
| `color` | - | STRING | 1.0 | 颜色格式 | 图像颜色，支持灰度值 (0.0-1.0)、十六进制 (#RRGGBB) 和 RGB (R,G,B) 格式 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 生成的纯色图像 |
| `mask` | MASK | 对应的遮罩图像 |