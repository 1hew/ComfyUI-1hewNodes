# Workflow Name - 当前工作流路径/名称。

## 典型用法

- 含日期的完整路径：保持 `full_path=True`、`strip_extension=True`、`date_format=yyyy-MM-dd`，便于将结果归档到日期目录下。
- 仅名称：设置 `full_path=False` 以获得纯文件名，同时使用 `prefix`/`suffix` 做定制。
- 保留扩展：当下游需要 `.json` 时设置 `strip_extension=False`。

## 注意与建议

- 请确保监控脚本将最新工作流路径写入 `current_workflow.tmp`，以便节点读取。
- 当 `date_format=none` 时输出不含日期目录，其他命名控制仍然生效。