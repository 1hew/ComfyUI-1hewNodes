# URL to Video - URL转视频

**节点功能：** `URL to Video`节点将视频URL转换为ComfyUI VIDEO对象，支持同步和异步下载方法，具备完善的错误处理和超时控制功能。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `video_url` | 必选 | STRING | "" | - | 视频文件的URL地址 |
| `timeout` | - | INT | 30 | 5-300 | 下载超时时间（秒） |
| `use_async` | - | BOOLEAN | False | True/False | 是否使用异步下载（推荐用于大文件） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `video` | VIDEO | ComfyUI VIDEO对象（当VIDEO类型可用时） |
| `error_message` | STRING | 错误信息（当VIDEO类型不可用时） |