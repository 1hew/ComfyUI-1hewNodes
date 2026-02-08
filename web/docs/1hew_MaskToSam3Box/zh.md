# Mask to SAM3 Box - 遮罩转 SAM3 Box 提示

**节点功能：** `Mask to SAM3 Box` 将输入遮罩解析为边界框提示，并输出 `SAM3_BOXES_PROMPT` 结构，适用于需要框提示的 SAM3 工作流。支持正/负提示、合并输出与按连通域拆分输出。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `mask` | - | MASK | - | - | 输入遮罩（支持批量）。遮罩值按 `> 0.5` 视为前景 |
| `condition` | - | COMBO | `positive` | `positive` / `negative` | 提示标签；`positive` 表示前景框，`negative` 表示排除框 |
| `output_mode` | - | COMBO | `merge` | `merge` / `separate` | `merge` 输出单个合并框；`separate` 按连通域输出多个框 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `sam3_box` | SAM3_BOXES_PROMPT | SAM3 框提示；单张遮罩输出字典，批量遮罩输出字典列表 |

## 功能说明

- 框提取：
  - `merge`：对全部前景像素取一个整体最小外接矩形。
  - `separate`：对前景连通域逐个计算外接矩形。
- 归一化格式：每个框输出为 `[cx, cy, bw, bh]`，均为 0-1 归一化值（相对于遮罩宽高）。
- 标签输出：`labels` 为布尔列表，`positive` 对应 `True`，`negative` 对应 `False`，长度与 `boxes` 一致。

## 典型用法

- 将分割/检测得到的 `MASK` 连接到本节点，再将 `sam3_box` 输出连接到需要 SAM3 框提示的节点。
- 需要排除区域时设置 `condition=negative`；希望减少框数量时使用 `output_mode=merge`。

## 注意与建议

- 遮罩噪声会导致 `separate` 生成大量小框，建议在上游做适度的形态学清理或改用 `merge`。
