# Match Brightness Contrast - 匹配亮度与对比度

**节点功能：** `Match Brightness Contrast` 节点用于将 `source_image` 的亮度与对比度映射到 `reference_image` 的色调分布。支持直方图匹配与均值/标准差匹配，并提供边缘区域统计与序列一致性策略，适合批处理与视频帧序列。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `source_image` | 必需 | IMAGE | - | - | 需要调整的源图像批次 |
| `reference_image` | 必需 | IMAGE | - | - | 提供目标色调分布的参考图像批次 |
| `edge_amount` | - | FLOAT | 0.2 | 0.0-8192.0 | 统计区域控制。`<=0`：全图；`<1.0`：按短边比例取边缘宽度；`>=1.0`：按像素取边缘宽度 |
| `consistency` | - | COMBO | `lock_first` | `lock_first` / `lock_mid` / `lock_end` / `frame_match` | 序列一致性。`lock_*` 从选定帧对计算一次参数并应用到全部源帧；`frame_match` 逐帧计算 |
| `method` | - | COMBO | `histogram` | `standard` / `histogram` | 匹配方法。`histogram` 通过通道 CDF 映射；`standard` 通过均值/标准差线性映射 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 调整后的图像批次 |

## 功能说明

- **匹配算法**：
  - `histogram`：基于通道直方图的累积分布函数 (CDF) 进行映射，能更精确地还原参考图的色调分布。
  - `standard`：基于均值和标准差的线性变换，保留更多源图的纹理特征，效果较柔和。
- **区域控制 (`edge_amount`)**：
  - 当值为 0 时，统计全图信息。
  - 当值大于 0 时，仅统计图像四周边缘区域的信息，忽略中心内容。这对于忽略主体差异、仅匹配环境氛围非常有用。
- **序列一致性 (`consistency`)**：
  - `lock_first`：使用 `source_image[0]` 与 `reference_image[0]` 计算参数。
  - `lock_mid`：使用输入批次的中间帧计算参数。
  - `lock_end`：使用输入批次的末帧计算参数。
  - `frame_match`：逐帧计算参数（参考帧按 `reference_image[i % ref_batch]` 取值）。

## 典型用法

- **视频色调统一**：连接视频帧到 `source_image`，参考序列到 `reference_image`，设置 `consistency=lock_mid` 或 `lock_end`，在参考序列前段存在偏移时保持映射稳定。
- **图像合成融合**：在图像合成时，将前景图作为 `source_image`，背景图作为 `reference_image`，设置适当的 `edge_amount` (如 0.2)，使前景边缘的色调与背景融合。

## 注意与建议

- 序列流程使用 `lock_*` 可将单一映射应用到全部源帧，有助于保持色调一致。
- `edge_amount` 将统计集中在边缘区域，可用于背景/环境氛围的稳定匹配。
