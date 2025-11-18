# Image Batch Group - 带重叠与填充的窗口分组

**节点功能：** `Image Batch Group` 将图像批次按固定窗口大小 `batch_size` 进行分组，支持设置窗口间 `overlap` 重叠，并可控制最后窗口的处理策略（丢弃、不满保留、回溯对齐末尾、颜色填充）。同时输出每组起始索引、窗口大小与有效帧数。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入图像批次 |
| `batch_size` | - | INT | 81 | 1-1024 | 每组窗口大小 |
| `overlap` | - | INT | 0 | 0-1024 | 相邻窗口的重叠帧数 |
| `last_batch_mode` | - | COMBO | `backtrack_last` | `drop_incomplete` / `keep_remaining` / `backtrack_last` / `fill_color` | 最后窗口处理策略 |
| `color` | - | STRING | `1.0` | 灰度/HEX/RGB/名称 | 当使用 `fill_color` 时的填充颜色；支持 `0.0-1.0`、`R,G,B`、HEX 与名称（如 `red`、`white`） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | `fill_color` 模式下返回已填充的批次；否则返回原始帧 |
| `group_total` | INT | 分组数量 |
| `start_index` | LIST(INT) | 每组的起始索引 |
| `batch_count` | LIST(INT) | 每组窗口大小（根据策略调整后） |
| `valid_count` | LIST(INT) | 每组真实帧数量（考虑填充影响） |

## 功能说明

- 参数校验：在 `backtrack_last` 模式下允许 `overlap <= batch_size`；其他模式要求 `overlap < batch_size`。
- 起始索引：步长为 `batch_size - overlap`；当步长不正时进行安全修正。
- 模式说明：
  - `drop_incomplete`：丢弃最后不满的窗口。
  - `keep_remaining`：保留最后的不满窗口。
  - `backtrack_last`：回溯调整最后窗口起点，使最后一窗完整对齐末尾。
  - `fill_color`：用纯色帧填充，使所有窗口满大小；输出包含填充后的批次。
- 颜色解析：支持灰度浮点、`R,G,B`（0–1 自动转换）、HEX、单字母缩写（`r/g/b/c/m/y/k/w`）与常见颜色名。
- 有效帧数：最后一组会反映真实（非填充）帧数量。

## 典型用法

- 构建带重叠的滑动窗口：设置 `batch_size=M`、`overlap=K` 用于序列处理。
- 仅保留完整窗口：设置 `last_batch_mode=drop_incomplete`。
- 末窗对齐尾部：设置 `last_batch_mode=backtrack_last`。
- 统一窗口尺寸：设置 `last_batch_mode=fill_color` 并指定 `color`。

## 注意与建议

- 使用 `fill_color` 时，如存在 Alpha 通道，其值固定为 `1.0`；灰度填充取 RGB 的平均值。
- 可通过 `valid_count` 区分真实与填充帧，特别是最后一组。