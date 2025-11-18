# Image HL Freq Combine - 高频/低频重组

**节点功能：** `Image HL Freq Combine` 将高频图与低频图按 `rgb`、`hsv` 或 `igbi` 三种方法进行重组，并提供 `high_strength` 与 `low_strength` 的强度控制，同时自动对齐批次。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `high_freq` | - | IMAGE | - | - | 高频层批次 |
| `low_freq` | - | IMAGE | - | - | 低频层批次 |
| `method` | - | COMBO | `rgb` | `rgb` / `hsv` / `igbi` | 重组方法 |
| `high_strength` | - | FLOAT | 1.0 | 0.0–2.0 | 高频层强度；`rgb/hsv` 在 0.5 中心附近偏移（`(high-0.5)*s+0.5`）|
| `low_strength` | - | FLOAT | 1.0 | 0.0–2.0 | 低频层强度 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 重组后的图像批次 |

## 功能说明

- 强度塑形：`rgb/hsv` 的高频在 0.5 中心附近进行偏移缩放；`igbi` 直接按比例缩放。