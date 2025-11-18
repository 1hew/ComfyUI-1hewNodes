# Mask Paste by BBox Mask - 依据 BBox 粘贴遮罩

**节点功能：** `Mask Paste by BBox Mask` 将 `paste_mask` 按 `bbox_mask` 定义的边界框粘贴到可选的 `base_mask` 上（缺省时使用与 `bbox_mask` 相同尺寸的全零遮罩）。支持批次循环，并将粘贴遮罩缩放到 bbox 尺寸。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `paste_mask` | - | MASK | - | - | 待粘贴的遮罩 |
| `bbox_mask` | - | MASK | - | - | 非零区域定义边界框 |
| `base_mask` | 可选 | MASK | - | - | 目标遮罩；缺省为与 `bbox_mask` 尺寸一致的全零 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `mask` | MASK | 在 bbox 区域粘贴后的遮罩 |

## 功能说明

- 批次循环：对 `base_mask`、`paste_mask`、`bbox_mask` 批次不一致时按索引取模对齐。
- 边界框检测：以阈值 `>10`（`[0..255]`）从 `bbox_mask` 提取 bbox。
- 尺寸适配：将 `paste_mask` 用 Lanczos 缩放到 bbox 尺寸并粘贴到相应位置。
- 兜底行为：当未检测到 bbox 时，返回原 `base_mask` 项。

## 典型用法

- 区域替换：在检测到的 bbox 内替换或插入精细化的遮罩区域。
- 裁切-粘贴流程：与遮罩裁切节点配合，实现遮罩内容的变换与重定位。

## 注意与建议

- 请确保 `paste_mask` 与 `base_mask` 的语义一致（白=选中），避免出现反相。
- 为精确对齐，建议预先准备与 bbox 内容匹配的 `paste_mask`（例如来自裁切结果）。