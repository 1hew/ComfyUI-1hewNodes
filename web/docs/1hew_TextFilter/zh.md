# Text Filter - 文本过滤。

## 典型用法

- 提炼提示词：设置 `filter_comment=True`、`filter_empty_line=True` 以获得紧凑、清爽的文本。
- 保留结构：当空行有语义时设置 `filter_empty_line=False`，同时去除注释。

## 注意与建议

- 三引号包裹的段落在启用注释过滤时视为注释并被排除。
- 在下游拼接处理前先进行过滤，可获得更稳定、易读的输入文本。