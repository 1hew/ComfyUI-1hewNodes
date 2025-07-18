# Text Join Multi - 文本连接多输入

**节点功能：** `Text Join Multi` 节点用于将多个文本输入连接成一个字符串，支持自定义分隔符和动态输入引用，具备智能注释过滤功能，常用于文本合并和格式化处理。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `text1` | 必选 | STRING | "" | 多行文本 | 第一个文本输入，支持多行文本和注释 |
| `text2` | 必选 | STRING | "" | 多行文本 | 第二个文本输入，支持多行文本和注释 |
| `text3` | 必选 | STRING | "" | 多行文本 | 第三个文本输入，支持多行文本和注释 |
| `text4` | 必选 | STRING | "" | 多行文本 | 第四个文本输入，支持多行文本和注释 |
| `text5` | 必选 | STRING | "" | 多行文本 | 第五个文本输入，支持多行文本和注释 |
| `separator` | - | STRING | "\\n" | - | 文本连接分隔符，默认为换行符 |
| `input` | 可选 | STRING | "" | - | 动态输入值，可在文本中使用 {input} 引用 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `string` | STRING | 连接后的文本字符串，已自动过滤注释内容 |

## 功能说明

### 注释处理
- **行首注释**：以 `#` 开头的整行注释会被完全移除
- **行内注释**：行中 `#` 后的内容会被移除，保留前面的有效内容
- **多行注释**：`"""..."""` 和 `'''...'''` 注释会被完全移除，不产生多余空行
- **空行保留**：保留原有文本的空行结构，仅移除注释产生的多余空行