# Multi Dict - 合并字典并追加键值

**节点功能：** `Multi Dict` 先按端口顺序合并动态 `dict_1..N`，再用 `key` / `value` 添加或覆盖一个键。后出现的同名键覆盖前面的；`key` / `value` 最后应用，因此永远覆盖合并结果。`value_type` 决定 `value` 如何转换。输出 `dict` 对象和 `json` 文本。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `value_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `dict` / `list` / `json` / `null` | 只作用于 `value` 的类型转换；见下表。 |
| `dict_1`、`dict_2`… | 可选 | * | - | - | 待合并的字典，按端口编号顺序合并；只接受字典，列表请拆成多个 `dict_N` 输入。 |
| `key` | 可选 | * | - | - | 最后添加/覆盖的键；接字符串或标量，数字/布尔会转成字符串；未连接或为空时只合并字典。 |
| `value` | 可选 | * | - | - | 最后添加/覆盖的值；先按 `value_type` 转换；`key` 为空时忽略；未连接时输出 `null`。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `dict` | DICT | 合并/追加后的字典，可继续接下一个 `Multi Dict` 的 `dict_1`，或接 `Save Text`。 |
| `json` | STRING | 紧凑 JSON 文本，可接 `Save Text`、LLM 或其它文本节点。 |

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
| `json` | 把字符串按 JSON 解析；对象→dict、数组→list、数字/布尔/null 对应 | `'{"a":1}'` → 对象 |
| `null` | 强制输出 `null` | 任意值 → `null` |

- `value_type` 只影响 `value`；`key` 始终按字符串/标量转字符串。
- `value` 未连接时始终输出 `null`，不受 `value_type` 影响；连接了但转换失败会直接报错。
- `json` 是严格解析：`Alice` 这种非 JSON 文本会报错；要保留普通字符串请用 `auto` 或 `string`。

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
- `key` 未连接或只有空白时，节点只做字典合并；此时 `value` 即使接了也会被忽略并打印提示。
- `value` 未连接但 `key` 有效时，输出 `{key: null}`。
- 字符串值不会被无辜包装：`auto` 下接一个字符串就是字符串，不会变成 `["字符串"]`；要数组请接一个真正的 list 值（例如 `Multi List` 的 `list` 输出）。
- 只接受 JSON 安全值：set、bytes、tensor、NaN / Infinity 会带路径报错，不会被悄悄转成字符串。

## 典型用法

- 单键对象：`key=<String name>`、`value=<String Alice>` → `{"name": "Alice"}`。
- JSON 文本转字典：`value=<String '{"a":1}'>`、`value_type=json` → `{"k": {"a": 1}}`。
- 数字文本转数字：`value=<String "30">`、`value_type=int` → `{"k": 30}`。
- 合并对象：`dict_1=<Multi Dict A>`、`dict_2=<Multi Dict B>` → 两个字典按顺序合并。
- 链式构建：`D2(dict_1=D1, key="b", value=2)` 得到 `{...D1, "b": 2}`。
- 保存：`dict` 或 `json` 输出接 `Save Text`，`format=json`。

## 注意

- `key` / `value` 都是连线端口，没有文本框；字面量需要由 `String` / `Int` / `Float` / `Boolean` 节点提供。
- 前端会把 `key` / `value` 固定在所有动态 `dict_N` 端口之后，保持“先合并、后追加”的视觉顺序。
- JSON 的键必须是字符串；空键只做合并。
