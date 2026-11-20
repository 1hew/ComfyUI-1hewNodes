# Text Path Extract - 文本路径取值

**节点功能：** `Text Path Extract` 按路径表达式从 `text` 输入框的内容中取值。内容可以是 JSON 文本、Markdown 代码块包裹的 JSON、Python 字面量、宽松 JSON（单引号、尾随逗号、True/False/None、无引号键）或普通文本。默认路径 `values[*]` 会把整段内容按值展开；用 `[0].content`、`[-1].bbox[0]`、`data.items[0].zh`、`values[0].ids`、`keys[*]` 这样的路径即可取到对应内容。

节点只负责取值：`value` 永远是一个列表，命中几项就有几项。把列表渲染成文字是下游的事，`List Text`、`Text List to String`、`Save Text` 各自按需要排版。

## 输入

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | ------ | -------- | ---- |
| `text` | STRING | `` | 多行 | 要解析的内容：JSON / 类 JSON / 普通文本 |
| `json_path` | STRING | `values[*]` | - | JSON 取值路径；`values[*]` 表示整段内容的所有值，空字符串表示整段内容本身。命名为 `json_path` 而非 `path`，避免与文件路径混淆 |
| `output_type` | COMBO | `list` | `list` / `json` | `list` 输出原生列表（保持值类型）；`json` 输出一个紧凑 JSON 数组文本 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `value` | ANY（列表） | 每个匹配一项的列表；命中 1 项时也是列表，未命中或解析失败时为 `[]`。`output_type=list` 时保持原值类型，`json` 时是紧凑 JSON 数组文本 |
| `count` | INT | 路径命中的项数；路径以 `.count` / `.length` / `[#]` 结尾时为该数量；未命中或输入为空时为 0。执行后输出名前面会显示这个数字（如 `3 count`） |

## 路径语法

| 写法 | 含义 |
| ---- | ---- |
| `[0]` / `[-1]` | 下标，同 Python：`0` 是第一项、`-1` 是最后一项；数组取元素，对象取其第 n 个值（按插入顺序，键名未知也能取） |
| `.content` / `content` | 字典键，开头的点可省略 |
| `[0]["content"]` / `[0].'content'` | 带引号的键，键名含空格或点号时使用；数字键也靠引号区分（`["0"]` 是键 `"0"`，`[0]` 是位置） |
| `keys[*]` / `keys[0]` | 对象的键名：全部 / 第 n 个；数组没有键名，取不到（要 `id` 用 `values[*].id`） |
| `values[*]` / `values[0]` | 对象的值：全部 / 第 n 个，默认路径就是 `values[*]`；数组上就是所有元素 / 第 n 个元素 |
| `values[*].content` | 通配：对每个值继续取 `content` |
| `$.data.items[0].zh` | 可选的 JSONPath 风格根 `$` |
| `values[*].content.count` / `[0].bbox.length` / `[0].bbox[#]` | 计数：多项匹配时为匹配数量，单个匹配时为该列表 / 字典 / 字符串的长度 |
| 空字符串 | 返回整段内容本身 |

`keys` / `values` 也可以写成单数 `key` / `value`，两者等价；遇到同名字段时以字段为准（见注意事项）。

全角的 `【】`、`“”`、`‘’` 会被自动规范化，中文输入法直接写 `【0】.“content”` 也能用。

## 功能说明

- 只取值不排版：`value` 输出原始列表，格式交给下游节点，同一份结果可以同时接到 `List Text` 加前缀、`Save Text` 存文件、`List Math Expression` 做计算。
- 多级解析：严格 JSON → Python 字面量 → 宽松 JSON（尾随逗号、True/False/None）→ 无引号键的宽松解析 → 截断内容的键值兜底。
- 代码块清理：自动剥离 Markdown 代码块围栏以及 JSON 前后的说明文字。
- 双层 JSON 解码：内容本身是 `"{\"a\": 1}"` 这样的字符串时，会再解一层。
- 精确键名：路径里的键区分大小写，不做同义词猜测；需要同义词兼容时请用 `Text Custom Extract`。
- 键值选择器：`keys[*]` / `values[*]` 分别取对象的所有键名和所有值，后面可以继续接 `[n]`、`values[*]`、`.字段`——键名不确定时写 `values[0].ids`，想确认命中的是哪个键就看 `keys[0]`；数组上 `values[*]` 就是逐个元素（数组没有键名，`keys` 只在对象上有效）。
- 计数：`count` 输出始终是路径命中的项数；路径末尾写 `.count` / `.length` / `.size` / `[#]` 则把数量本身作为值返回，此时 `count` 输出也是这个数量。
- 前端数值：执行完成后 `count` 输出名前面会显示数字（如 `3 count`），与 `count` 通道值一致；重新连线会先恢复成纯 `count`，下次执行再刷新。

## 典型用法

- 取第一项文本：`json_path=[0].content`，`value` 是 `["犀牛"]`。
- 取所有 content：`json_path=values[*].content`，`value` 是 `["犀牛","犀牛","犀牛"]`。
- 取所有 bbox：`json_path=values[*].bbox`，`value` 是 `[[184,410,845,716],[508,679,1000,953],[690,264,992,588]]`。
- 取数量：直接连 `count` 输出，或把路径写成 `values[*].content.count`。
- 参考图的键名不确定（可能是 `reference_image_1`，也可能是 `image_1`）：`json_path=values[0].ids` 直接取第一张图的标注；多张并存时 `values[0]` / `values[-1]` 取第一 / 最后一张，`values[*].ids` 取全部，`keys[0]` 能看出命中的是哪个键。
- 每个索引加 1 再写回：`value` → `List Math Expression`（`a + 1`）→ `Text Path Replace`。
- 把列表渲染成文字：接 `List Text`（逐项加前后缀）或 `Text List to String`（按分隔符拼接）。
- 存成 JSON 文件：接 `Save Text`，`format` 选 `json`。

## 注意与建议

- `text` 为空或全是空白时返回空列表，`count` 为 0，不会报错。
- 路径未命中同样返回空列表；下游按空列表处理即可。
- 解析器遇到无法处理的负载而抛错时返回空列表、`count` 为 0（错误打印到控制台），下游不会收到端口未声明的类型。
- 普通文本（不是 JSON）没有“项”可遍历，`values[*]` 取不到东西；这时把 `json_path` 留空即可原样取到整段文本。
- 命中 1 项时 `value` 也是列表（`[值]`），不是裸值；需要单个值就用 `List Text` 或索引取第一项。
- 键名本身叫 `count` 时不会冲突：对象里确实存在该键则优先返回键值；想强制按字面键取值就写成 `["count"]`。`keys` / `values`（含单数写法）同理，payload 里真有 `values` 字段时以字段为准，写 `["values"]` 才是字面键。
- `[0]` 在对象上取第 n 个值，但真实键优先：`{"0": …}` 里 `[0]` 拿到的是键 `"0"`；要纯按位置取就写 `values[0]`，它先取值列表、再做列表下标，不会再被键名干扰。
- 列表没有键名：`keys[*]` 在数组上取不到（返回空列表）；要 `id` 就写 `values[*].id`（`box_001`…），要每个元素的字段名写 `values[*].keys[*]`。
- 对象位置依赖插入顺序（`json.loads` 保留文本顺序）：上游若改为排序序列化，`[0]` 可能指向别的项；需要稳定就写键名，或用 `keys[*]` 核对 / `values[*]` 全取后自行筛选。
- 数字显示依赖前端扩展 `web/js/output_label.js`，未加载该扩展时只是输出名不显示数字，取到的值不受影响。
