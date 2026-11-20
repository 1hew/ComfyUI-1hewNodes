# Text Path Replace - 按路径写回替换值

**节点功能：** `Text Path Replace` 把值写回 JSON 类文本中路径匹配到的位置。它是 `Text Path Extract` 的写入部分：读取时用的路径，同时也决定新值写到哪里。计算放在上游完成（`List Math Expression`、`Math Expression`、`Multi Dict`、字符串节点等），本节点只负责放置替换值。

## 输入

| 参数名称 | 插槽 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | ---- | -------- | ------ | -------- | ---- |
| `text` | - | STRING | `` | 多行 | 要编辑的 JSON、代码块 JSON 或 Python 字面量 |
| `path` | - | STRING | `values[*]` | - | 写入位置的路径，语法与 `Text Path Extract` 相同 |
| `replace` | - | *（列表） | - | - | 要写入的值。列表按顺序使用；单个值广播到所有匹配位置 |
| `value_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `dict` / `list` / `json` / `null` | 对 `replace` 的每一项做类型转换，含义与 `Multi Dict` 相同 |
| `output_format` | - | COMBO | `pretty` | `pretty` / `compact` | `text` 的排版方式 |

## 输出

| 输出名称 | 数据类型 | 描述 |
| -------- | -------- | ---- |
| `text` | STRING | 编辑后的文档，缩进或紧凑排版 |
| `value` | * 列表 | 编辑后的文档装在只有一项的列表中，根节点是数组时下游也不会被拆开 |
| `count` | INT | 实际写入的位置数；路径未命中时为 0 |

## 替换规则

| `replace` 项数 | 匹配数 | 结果 |
| -------------- | ------ | ---- |
| 1 | N | 该值写入全部 N 个位置 |
| N | N | 按匹配顺序一一替换 |
| 其他数量 | N | 报错：数量不一致，不写入任何位置 |

## 不能写入的路径

- `keys[*]` / `key[0]`：键名只能读取，不能用于替换，会被拒绝。
- `count` / `length` / `size` / `[#]` 运算：只读，会被拒绝。
- 只写 `values` 而不指定位置或 `[*]`：整组没有单一位置，会被拒绝。

## 典型用法

- 每个索引加 1：`Text Path Extract`（`layers[*].index`）→ `List Math Expression`（`a + 1`）→ `Text Path Replace`（`path = layers[*].index`，`replace` 接数学节点的输出）。
- 把所有 content 替换为同一个字符串：`path = [*].content`，`replace` 只接一个值即可广播。
- 替换嵌套的坐标：`path = [1].bbox[0]`，`replace = [50]`。

## 注意事项

- 输入文本不会被原地修改，节点在副本上操作并返回编辑后的文档。
- 路径未命中时，`text` 原样返回，`count` 为 0。
- 数量不一致或路径无效会报错，不会悄悄写到别处。
- 无法表示为 JSON 的值（例如 Python 的 set）会被拒绝并报错。
