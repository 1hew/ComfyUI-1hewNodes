# Detect Remove BG Refine - RMBG 掩码后处理细化

**节点功能：** `Detect Remove BG Refine` 作为 `RMBG-1.4` 等抠图结果的后处理节点使用。输入**原始图像**与模型输出 `mask`，执行 alpha 细化、边缘抗锯齿、可选矢量硬边与去色边，输出优化后的 RGBA 与 refined mask。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必填 | IMAGE | - | - | 原始图像（必须是原图，不是模型处理后的图） |
| `mask` | 必填 | MASK | - | - | RMBG 模型输出的 alpha mask |
| `type` | - | COMBO | `bitmap` | `bitmap` / `vector` | 边缘模式：位图柔边或矢量硬边 |
| `subject_protect` | - | FLOAT | 0.85 | 0.0~1.0 | 主体保护强度，越高越倾向保留核心前景 |
| `feather` | - | FLOAT | 1.0 | 0.0~64.0 | 羽化控制（内部映射为平滑强度） |
| `decolor_edge` | - | FLOAT | 1.0 | 0.0~1.0 | 边缘去色边强度 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 普通 RGBA 图像（最后通道为 refined alpha，不做 premultiply） |
| `mask` | MASK | 细化后的 alpha mask |

## 功能说明

- Alpha refine：先做双边/高斯平滑、核心前景保护与边缘抗锯齿。
- `vector` 模式：在细化后执行硬边轮廓重建，适合图形化边缘需求。
- 固定背景估计策略：白/黑优先判定，失败后自动回退边缘统计。
- 去色边（decontaminate）：始终基于原图 RGB + refined alpha 进行恢复。

## 典型用法

- RMBG 结果精修：`Detect Remove BG` 输出 `mask` 接本节点，`image` 使用原始输入图。
- LOGO/插画类硬边：`type=vector`，适当提高 `subject_protect`。
- 人像柔和边缘：`type=bitmap`，`feather` 与 `decolor_edge` 适中调整。

## 注意与建议

- 请勿把“模型处理后的图像”再送入本节点，会放大边缘污染与色偏。
- 本节点固定内部策略，不提供 `bg_mode`、`min_area_ratio`、`premultiply` 额外参数。
