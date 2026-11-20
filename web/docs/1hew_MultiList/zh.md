# Multi List - 把多个元素合并成列表

**节点功能：** `Multi List` 按端口顺序把动态 `any_1..N` 合并成一个 JSON 数组。`extend` 对应 Python 的 `list.extend`：把列表/元组输入摊平一层；`append` 对应 `list.append`：每个输入作为一个元素，列表保持嵌套。输出只有一个 `data`，类型由 `output_type` 决定（`list` 或紧凑 `json`）。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `mode` | - | COMBO | `extend` | `extend` / `append` | `extend` 摊平列表/元组一层；`append` 保留嵌套。 |
| `any_1`、`any_2`… | 可选 | * | - | - | 待合并的元素，按端口编号顺序；后续通过动态端口接入。 |
| `output_type` | - | COMBO | `list` | `list` / `json` | `list` 输出原生数组，`json` 输出紧凑 JSON 文本。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `data` | * | 合并后的数组；`output_type=json` 时是紧凑 JSON 文本。可继续接入另一个 `Multi List` 或 `Multi Dict`。 |

## `mode` 语义

| 输入 `"a"`、`[1, 2]`、`{"b": 3}` | 结果 |
| ---------------------------------------- | ---- |
| `extend` | `["a", 1, 2, {"b": 3}]` |
| `append` | `["a", [1, 2], {"b": 3}]` |

## 功能说明

- `extend` 只摊平 `list` / `tuple`：字符串不会被拆成字符，字典不会被拆成键（在 JSON 里它们是值，不是元素组）。
- `append` 永远只加一个元素；列表会作为嵌套数组保留。
- 只摊平一层：嵌套列表仍作为元素保留，需要更深展开时串联多个节点。
- 单值不包装：接单个字符串、数字或字典时就是一个元素，不会自动变成数组。
- 未连接或 `None` 的端口跳过；全部为空时输出 `[]`。
- 值约束：set、bytes、tensor、NaN / Infinity 等非 JSON 值会在序列化时带路径报错。

## 典型用法

- 把多个标量、字典或列表拼成一个数组。
- 把多个 `Multi Dict` 的输出拼成对象数组。
- 把上游单个 list 值展开（`extend`）或整体作为一个嵌套元素（`append`）。

## 注意

- `any_N` 接的是元素；要合并字典请用 `Multi Dict`，不要接 `Multi Dict` 的 `dict` 到 `any_N` 再指望合并。
- 输出 `list` 是一个数组值，不是 ComfyUI 的 `OUTPUT_IS_LIST`；接 `Multi Dict` 的 `value` 会作为数组值。
