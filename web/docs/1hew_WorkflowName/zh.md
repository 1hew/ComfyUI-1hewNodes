# 工作流名称

**节点功能：** `Workflow Name` 节点通过监控临时文件自动获取当前工作流文件名，支持路径控制、自定义前缀后缀和日期格式化，常用于动态工作流命名和文件组织。

## 输入参数

| 参数名 | 必需 | 数据类型 | 默认值 | 范围 | 描述 |
|--|--|--|--|--|--|
| `prefix` | 可选 | STRING | "" | - | 添加到工作流名称前的自定义前缀 |
| `suffix` | 可选 | STRING | "" | - | 添加到工作流名称后的自定义后缀 |
| `date_format` | 可选 | COMBO[STRING] | "yyyy-MM-dd" | 多种格式 | 日期格式前缀：none, yyyy-MM-dd, yyyy/MM/dd, yyyyMMdd, yyyy-MM-dd HH:mm, yyyy/MM/dd HH:mm, yyyy-MM-dd HH:mm:ss, MM-dd, MM/dd, MMdd, dd, HH:mm, HH:mm:ss, yyyy年MM月dd日, MM月dd日, yyyyMMdd_HHmm, yyyyMMdd_HHmmss |
| `full_path` | 可选 | BOOLEAN | True | True/False | 是否包含完整路径（相对于工作流目录）还是仅文件名 |
| `strip_extension` | 可选 | BOOLEAN | True | True/False | 是否从文件名中移除 .json 扩展名 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|-------------|-----------|-------------|
| `string` | STRING | 应用格式化选项后的处理后工作流名称 |
