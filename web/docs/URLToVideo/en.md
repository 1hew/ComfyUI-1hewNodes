# URL to Video

**Node Function:** The `URL to Video` node converts video URLs to ComfyUI VIDEO objects, supporting both synchronous and asynchronous download methods with comprehensive error handling and timeout control.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `video_url` | Required | STRING | "" | - | Video file URL address |
| `timeout` | - | INT | 30 | 5-300 | Download timeout in seconds |
| `use_async` | - | BOOLEAN | False | True/False | Whether to use asynchronous download (recommended for large files) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `video` | VIDEO | ComfyUI VIDEO object (when VIDEO types available) |
| `error_message` | STRING | Error message (when VIDEO types unavailable) |