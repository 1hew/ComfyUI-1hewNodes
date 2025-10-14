# Image Batch Group - 图像批次分组

**节点功能：** `Image Batch Group`节点用于将图像批次按指定大小分割成更小的组，支持重叠帧处理和智能填充策略。为 ComfyUI 中的顺序图像工作流提供灵活的批处理能力。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 需要分组的输入图像批次 |
| `batch_size` | 必选 | INT | 81 | 1-1024 | 每个输出批次的大小，步长：4 |
| `overlap` | 必选 | INT | 0 | 0-1024 | 连续批次间的重叠帧数，步长：1 |
| `last_batch_mode` | 必选 | COMBO[STRING] | keep_remaining | keep_remaining, backward_extend, append_image | 最后一批的处理策略 |
| `color` | 可选 | STRING | "1.0" | - | 填充图像的颜色规格 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 处理后的图像批次（根据模式可能包含填充图像） |
| `batch_total` | INT | 创建的批次总数 |
| `start_index` | INT | 每个批次的起始索引（列表输出） |
| `batch_count` | INT | 每个批次中的图像数量（列表输出） |
| `effective_count` | INT | 每个批次中有效（非重叠）图像的数量（列表输出） |