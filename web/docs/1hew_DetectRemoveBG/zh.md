# Detect Remove BG - 多后端抠图与遮罩输出

**节点功能：** `Detect Remove BG` 提供统一的去背景入口，支持 `RMBG-1.4`、`RMBG-2.0`、`birefnet`、`Inspyrenet` 等后端，并输出前景图与 alpha 遮罩。可选择直接输出 RGBA，或合成白/黑背景图。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必填 | IMAGE | - | - | 输入图像批次 |
| `model` | - | COMBO | `RMBG-1.4` | `none` / `RMBG-1.4` / `RMBG-2.0` / `birefnet-general` / `birefnet-general-lite` / `Inspyrenet` | 抠图模型后端选择 |
| `add_background` | - | COMBO | `alpha` | `alpha` / `white` / `black` | 输出图像背景模式 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 抠图结果：`alpha` 模式为 RGBA，`white/black` 模式为 RGB 合成图 |
| `mask` | MASK | 0~1 浮点 alpha 遮罩 |

## 功能说明

- 多模型统一接口：按 `model` 自动切换不同推理后端。
- 自动准备模型：部分模型会尝试下载到 `models/rembg` 并进行缓存复用。
- 经典回退模式：`model=none` 使用传统颜色差分方法估计 alpha。
- 背景合成可选：可直接得到透明背景（RGBA）或白/黑底图，便于后续节点接入。

## 典型用法

- 通用抠图：`model=RMBG-1.4`，`add_background=alpha`。
- 需要直接导出白底图：`add_background=white`。
- 调试对比不同模型效果：同一输入切换 `model` 多次比较边缘质量与主体保留。

## 注意与建议

- 不同模型依赖环境不同；若缺少依赖会在日志中提示并返回失败。
- `RMBG-2.0` 单文件模式依赖 `onnxruntime`；若未安装，节点会直接报错并提示在 ComfyUI 当前 Python 环境中安装对应包。
- 若你后续要做细化边缘和去色边，建议将 `mask` 接到 `Detect Remove BG Refine` 节点进一步处理。
