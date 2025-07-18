# List Custom Float - 自定义浮点数列表

**节点功能：** `List Custom Float`节点生成浮点数类型的列表，支持连字符分割和多种分隔符，用于灵活的文本到浮点数列表转换。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `custom_text` | - | STRING | "" | 多行文本 | 支持多种分隔符的自定义文本输入 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `float_list` | FLOAT | 生成的浮点数列表 |
| `count` | INT | 列表中浮点数的数量 |

## 功能说明

### 分隔符支持
- **连字符优先级**：当存在只包含连字符(---)的行时，只使用连字符分割，覆盖其他分隔符
- **多重分隔符**：支持逗号(,)、分号(;)和换行符(\n)分隔
- **中英文分隔符**：支持中文(，；)和英文(,;)标点符号