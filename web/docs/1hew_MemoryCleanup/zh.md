# Memory Cleanup - 内存清理

**节点功能：** `Memory Cleanup` 请求 ComfyUI 在当前任务结束后释放执行缓存与模型占用，适合多次排队执行之间清理 RAM/VRAM。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `anything` | 可选 | `*` | - | - | 任意输入，会被直接透传到输出。 |
| `unload_model` | - | BOOLEAN | `False` | `True` / `False` | 是否在当前任务结束后请求卸载已加载模型。开启更省 VRAM/RAM，但下次执行可能需要重新加载模型。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `output` | `*` | 透传 `anything` 输入的值。 |

## 功能说明

- 执行时调用 `gc.collect()` 触发 Python 垃圾回收。
- 调用 ComfyUI 的 `soft_empty_cache` 清理 PyTorch 缓存。
- 通过 prompt_queue 设置 `free_memory` 标志，请求释放执行缓存。
- 当 `unload_model=True` 时，额外设置 `unload_models` 标志，请求卸载已加载模型。

## 典型用法

- 在多次排队任务的间隙插入此节点，避免 VRAM 持续增长。
- 切换模型或工作流时开启 `unload_model=True`，确保旧模型被释放。
- 将该节点放在工作流末尾，确保每次执行后自动清理。

## 注意与建议

- 此节点为输出节点（`is_output_node=True`），ComfyUI 会确保它在工作流末端执行。
- 开启 `unload_model` 会导致下次执行重新加载模型，增加单次执行时间，建议按需使用。
- `anything` 输入支持任意类型，可串联前置节点以保证清理在执行顺序正确的位置触发。
