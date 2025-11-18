# Video Cut Group - 视频硬切检测与分组

**节点功能：** `Video Cut Group` 对一组视频帧进行硬切检测并分组。提供快速简化 SSIM 模式与多核高斯模糊 SSIM 检测，支持动态阈值、统一分组规则（`min_frame_count` / `max_frame_count`），并可通过 `add_frame` / `delete_frame` 进行人工调整。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | - | IMAGE | - | - | 输入视频帧（图像批次） |
| `threshold_base` | - | FLOAT | 0.8 | 0.0-1.0 | 基础阈值，比较规则为 `1-SSIM > threshold`；值越大越严格 |
| `threshold_range` | - | FLOAT | 0.05 | 0.01-0.2 | 在基础值附近构造范围，用于生成多个阈值 |
| `threshold_count` | - | INT | 2 | 1-10 | 在范围内使用的阈值数量 |
| `kernel` | - | STRING | `3, 7, 11` | 奇数尺寸 | 以逗号分隔的高斯核尺寸；要求为 `≥3` 的奇数 |
| `min_frame_count` | - | INT | 10 | 1-1000 | 最小分段长度；过近切点将被合并 |
| `max_frame_count` | - | INT | 0 | 0-10000 | 最大分段长度；为 `0` 时不拆分过长片段 |
| `fast` | - | BOOLEAN | `False` | - | 启用简化版 SSIM 快速检测 |
| `add_frame` | - | STRING | `` | 逗号列表 | 手动添加切点帧；兼容中/英文逗号 |
| `delete_frame` | - | STRING | `` | 逗号列表 | 手动删除切点帧（保留起始帧 `0`）；兼容中/英文逗号 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 每个分段的起始帧（关键帧）合集 |
| `group_total` | INT | 分段数量 |
| `start_index` | LIST(INT) | 每个分段的起始帧索引（始终包含 `0`） |
| `batch_count` | LIST(INT) | 每段的帧计数 |

## 功能说明

- 预处理：转换为 `float32`，必要时将 `[0,1]` 放大到 `[255]`，并统一转灰度。
- 多核检测：对相邻帧在不同核尺寸下计算模糊 SSIM；在多个阈值下进行检测。
- 快速模式：使用简化 SSIM 进行快速检测；随后统一应用分组规则。
- 统一融合：合并所有配置的检测切点；一致应用 `min_frame_count` 与 `max_frame_count`。
- 人工编辑：通过逗号索引添加或删除切点；起始帧 `0` 始终保留。
- 稳健输出：返回关键帧批次并附带分段元数据。

## 典型用法

- 场景切分：调整 `threshold_base` 与 `min_frame_count` 以在灵敏度与稳定性间平衡。
- 关键帧抽取：使用输出 `image`（各段起始帧）作为缩略图或锚点。
- 人工修正：通过 `add_frame` 与 `delete_frame` 进行细节修整。

## 注意与建议

- `threshold_base` 越大代表越严格的检测（切点更少），因比较采用 `1-SSIM > threshold`。
- 当 `max_frame_count > 0` 时，会将过长片段按该上限进行间隔拆分。