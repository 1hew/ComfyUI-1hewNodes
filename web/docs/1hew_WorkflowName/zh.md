# Workflow Name - 工作流名称

**节点功能：** `Workflow Name` 节点读取工作流监控脚本记录的当前工作流路径，并输出处理后的名称字符串。支持前后缀、日期目录格式、完整路径输出，以及扩展名处理选项。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `prefix` | - | STRING | - | - | 输出名称前缀。 |
| `suffix` | - | STRING | - | - | 输出名称后缀。 |
| `date_format` | - | COMBO | `yyyy-MM-dd` | `none` / `yyyy-MM-dd` / `yyyy/MM/dd` / `yyyyMMdd` / `yyyy-MM-dd HH:mm` / `yyyy/MM/dd HH:mm` / `yyyy-MM-dd HH:mm:ss` / `MM-dd` / `MM/dd` / `MMdd` / `dd` / `HH:mm` / `HH:mm:ss` / `yyyy年MM月dd日` / `MM月dd日` / `yyyyMMdd_HHmm` / `yyyyMMdd_HHmmss` | 日期目录格式，用于构建输出字符串。 |
| `full_path` | - | BOOLEAN | False | - | 为 True 时输出包含目录部分，并将路径分隔符统一为 `/`。 |
| `strip_extension` | - | BOOLEAN | True | - | 为 True 时移除 `.json` 扩展名。 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `string` | STRING | 处理后的工作流名称字符串。 |

## 功能说明

- 监控集成：读取工作流监控脚本生成的 `current_workflow.tmp`。
- 名称处理：提取文件名（或完整路径），应用 `prefix`/`suffix`，并按需处理 `.json` 扩展名。
- 日期目录：启用 `date_format` 时输出结构为 `date_str/原始工作流名/结果`。
- 稳定读取：在文件写入并发场景下进行多次尝试读取。

## 典型用法

- 作为保存节点的 `filename_prefix`：与图像/视频保存节点结合，实现按工作流与日期归档输出。
- 输出可复用命名：设置 `strip_extension=True` 并选择稳定的 `date_format`，获得结构化路径。

## 注意与建议

- 工作流监控脚本持续运行时，`current_workflow.tmp` 会保持更新。
- `full_path=True` 适合需要目录层级信息的归档策略。
