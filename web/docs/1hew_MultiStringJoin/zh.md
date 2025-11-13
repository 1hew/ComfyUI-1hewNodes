# Multi String Join - 动态字符串连接

**节点功能：** `Multi String Join` 节点用于将动态 `string_X` 文本输入拼接为一个字符串，支持逐输入的注释过滤、空行控制、`{input}` 占位符替换以及自定义分隔符。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `string_1` | 可选 | STRING | - | - | 动态文本输入首端口；可扩展 `string_2`、`string_3` … |
| `filter_empty_line` | 必选 | BOOLEAN | False | True/False | 在每个文本处理后移除空行 |
| `filter_comment` | 必选 | BOOLEAN | False | True/False | 过滤行内注释（`#`）与三引号注释块（`'''`/`"""`） |
| `separator` | 必选 | STRING | "\\n" | - | 拼接分隔符；支持 \\n、\\t、\\r 转义 |
| `input` | 可选 | STRING | "" | - | 动态输入占位；可在文本中使用 `{input}` 引用 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `string` | STRING | 拼接后的结果文本 |
