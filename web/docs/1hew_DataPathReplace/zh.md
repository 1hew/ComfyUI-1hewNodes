# Data Path Replace - 按路径写回替换值

**节点功能：** `Data Path Replace` 把值写回 JSON 类文本中路径匹配到的位置。它是 `Data Path Extract` 的写入部分：读取时用的路径，同时也决定新值写到哪里。要写入的值直接填在 `value` 里，按 `value_type` 转成目标类型，本节点只负责放置。

与 `Data Path Extract` 一样，本节点不做渲染：编辑后的文档通过 `data` 端口输出，形状与输入一致——字典还是字典、数组还是数组；排版交给下游（`Save Text` 写文件、`List Text` 加前后缀，或任意 JSON 处理节点）。

## 输入

| 参数名称 | 插槽 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | ---- | -------- | ------ | -------- | ---- |
| `data` | *（必填连线） | - | - | 要编辑的文档，连线输入：JSON 文本、代码块 JSON、Python 字面量，或已解析的 dict / list |
| `data_path` | - | STRING | `[*]` | - | 写入位置的路径，语法与 `Data Path Extract` 相同 |
| `value` | - | STRING | `` | - | 要写入的值，直接输入，按 `value_type` 转换。填单个值会广播到所有匹配位置；填 JSON 数组（如 `[1,2,3]`）则按顺序一一写入 |
| `value_type` | - | COMBO | `auto` | `auto` / `dict` / `list` / `string` / `int` / `float` / `boolean` / `null` | 作用于 `value`，含义与 `Multi Dict` 相同。`auto` 下文本就是文本，需要数字请选 `int` / `float` |

## 输出

| 输出名称 | 数据类型 | 描述 |
| -------- | -------- | ---- |
| `data` | ANY | 编辑后的文档，形状与输入一致：对象还是对象、数组还是数组。根节点是数组时会被 ComfyUI 下游逐项拆开，需要整体处理请接 `Multi List` |
| `count` | INT | 实际写入的位置数；路径未命中时为 0 |

## 替换规则

| `value` 项数 | 匹配数 | 结果 |
| ------------ | ------ | ---- |
| 1 | N | 该值写入全部 N 个位置 |
| N | N | 按匹配顺序一一替换 |
| 其他数量 | N | 报错：数量不一致，不写入任何位置 |

## 不能写入的路径

- `keys[*]` / `key[0]`：键名只能读取，不能用于替换，会被拒绝。
- `count` / `length` / `size` / `[#]` 运算：只读，会被拒绝。
- 只写 `values` 而不指定位置或 `[*]`：整组没有单一位置，会被拒绝。

## 典型用法

- 把所有 content 统一改成同一个值：`path = [*].content`，`value` 填一个新值，它会广播到每一处。
- 逐个写入不同的值：`path = layers[*].index`，`value` 填 `[1, 2, 3]`、`value_type` 选 `int`，三个位置依次写入。
- 替换嵌套的坐标：`path = [1].bbox[0]`，`value` 填 `50`、`value_type` 选 `int`。
- 保存结果：把 `data` 接到 `Save Text`，`format` 选 `json`。

## 注意事项

- 输入文本不会被原地修改，节点在副本上操作并返回编辑后的文档。
- 路径未命中时，`value` 是原样的文档，`count` 为 0。
- 数量不一致或路径无效会报错，不会悄悄写到别处。
- 无法表示为 JSON 的值（例如 Python 的 set）会被拒绝并报错。
- `value_type=auto` 会原样保留上游类型：字符串 `"2"` 写进去还是字符串。需要数字时选 `int` / `float`。
