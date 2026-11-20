# Multi Dict - 合并字典并追加键值

**节点功能：** `Multi Dict` 先按端口顺序合并动态 `dict_1..N`，再添加或覆盖一个键值对。键和值都直接输入（`key` / `value`），不需要连线；`key_type` 决定键的类型，`value_type` 决定值如何转换。两个输出同时可用：`dict` 输出原生对象，`json` 输出紧凑 JSON 文本。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `dict_1`、`dict_2`… | 可选 | * | - | - | 待合并的字典，按端口编号顺序合并；只接受字典，列表请拆成多个 `dict_N` 输入。 |
| `key` | - | STRING | `` | - | 最后添加/覆盖的键，直接输入；为空时只合并字典。JSON 对象的键只能是字符串。 |
| `key_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `null` | `auto` 像 Python 字面量一样推断（`1`→int、`1.5`→float、`true`→bool、`null`→None）；`string` 永远保持文本。其余选项强制该类型，转不了就报错。 |
| `value` | - | STRING | `` | - | 最后添加/覆盖的值，按 `value_type` 转换；`key` 为空时忽略；留空则写入 `null`。 |
| `value_type` | - | COMBO | `auto` | `auto` / `dict` / `list` / `string` / `int` / `float` / `boolean` / `null` | 作用于 `value` 的类型转换；见下表。两个输出都按该类型写出，便于互转。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `dict` | DICT | 合并后的字典，可接下一个 `Multi Dict` 的 `dict_1`。 |
| `json` | STRING | 紧凑 JSON 文本，可接 `Save Text` 或 LLM 文本节点。 |

## `value_type` 转换

| 取值 | 行为 | 示例 |
| ---- | ---- | ---- |
| `auto` | 原样保留，不做转换 | `"30"` → `"30"`；`{...}` → 对象 |
| `string` | 转成文本；dict/list 转紧凑 JSON 文本 | `30` → `"30"`；`{"a":1}` → `"{\"a\":1}"` |
| `int` | 转整数；字符串按数字解析，小数会截断 | `"30"` → `30`；`3.7` → `3`；`"3.7"` → `3` |
| `float` | 转浮点 | `"3.5"` → `3.5`；`2` → `2.0` |
| `boolean` | 转布尔；`true` / `1` / `yes` / `on` / 非零数字为真 | `"true"` → `true`；`"0"` → `false` |
| `dict` | 解析 JSON 文本并要求是对象 | `'{"a":1}'` → 对象；`[1,2]` 报错 |
| `list` | 解析 JSON 文本并要求是数组 | `'[1,2]'` → 数组；`{"a":1}` 报错 |
| `null` | 强制输出 `null` | 任意值 → `null` |

- `value_type` 只影响 `value`；`key` 由 `key_type` 处理。
- Python 字典接受任何可哈希的键，因此 `key_type` 产出的是真正的 int / float / bool / None 键。两个输出都合法：`dict` 保留这些键类型，`json` 只能写成字符串（`1` → `"1"`、`true` → `"true"`、`None` → `"null"`）。
- `value` 为空时始终写入 `null`，不受 `value_type` 影响；填了但转换失败会直接报错。
- `auto` 下文本就是文本（`"30"` 仍是 `"30"`）；需要数字请选 `int` / `float`。
- 两个输出始终同类型：`dict` 端口是原生 Python 值，`json` 端口是对应的 JSON 字面量；`dict` / `list` / `int` / `float` / `boolean` / `null` / `string` 都能原样互转。

## 顺序与覆盖

1. `dict_1` → `dict_2` → `dict_3`… 按端口编号顺序合并；
2. 同名键由后面的覆盖前面的；键保留第一次出现的位置；
3. 最后应用 `key` / `value`（先经过 `value_type`），所以它会覆盖所有字典里的同名键。

示例：

```text
dict_1 = {a: 1, b: 1}
dict_2 = {a: 2}
key = a, value = 9
结果 = {a: 9, b: 1}
```

## 功能说明

- 只接受字典：`dict_N` 接字符串、数字、列表都会报错；要合并列表里的多个字典，请把每个字典接到单独的 `dict_N` 端口。
- `key` 支持字符串、整数、浮点、布尔；接字典/列表会报错。
- `key` 为空或只有空白时，节点只做字典合并；此时 `value` 即使填了也会被忽略并打印提示。
- `value` 为空但 `key` 有效时，输出 `{key: null}`。
- 字符串值不会被无辜包装：`auto` 下接一个字符串就是字符串，不会变成 `["字符串"]`；要数组请接一个真正的 list 值（例如 `Multi List` 的 `list` 输出）。
- 只接受 JSON 安全值：set、bytes、tensor、NaN / Infinity 会带路径报错，不会被悄悄转成字符串。

## 典型用法

- 单键对象：`key=<String name>`、`value=<String Alice>` → `{"name": "Alice"}`。
- JSON 文本转字典：`value` 填 `{"a":1}`、`value_type=dict` → `{"k": {"a": 1}}`。
- 数字文本转数字：`value=<String "30">`、`value_type=int` → `{"k": 30}`。
- 合并对象：`dict_1=<Multi Dict A>`、`dict_2=<Multi Dict B>` → 两个字典按顺序合并。
- 链式构建：`D2(dict_1=D1, key="b", value=2)` 得到 `{...D1, "b": 2}`。
- 保存：`dict` 或 `json` 输出接 `Save Text`，`format=json`。

## 注意

- `key` / `value` 都是连线端口，没有文本框；字面量需要由 `String` / `Int` / `Float` / `Boolean` 节点提供。
- 前端会把 `key` / `value` 固定在所有动态 `dict_N` 端口之后，保持“先合并、后追加”的视觉顺序。
- JSON 的键必须是字符串；空键只做合并。
