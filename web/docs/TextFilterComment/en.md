# Text Filter Comment

**Node Function:** The `Text Filter Comment` node is used to filter out comment lines from text, supporting both single-line comments (starting with #) and multi-line comments (enclosed by triple quotes), while preserving non-comment blank lines.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `text` | Required | STRING | "" | Multi-line text | Input text to filter comments from |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Text with comments filtered out |