# Text Join Multi

**Node Function:** The `Text Join Multi` node merges up to five multi-line text inputs into a single string, supporting `{input}` placeholder substitution, comment filtering, empty-line control, and configurable separators.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `text1` | Required | STRING | "" | Multi-line text | First text input, processed line by line |
| `text2` | Required | STRING | "" | Multi-line text | Second text input |
| `text3` | Required | STRING | "" | Multi-line text | Third text input |
| `text4` | Required | STRING | "" | Multi-line text | Fourth text input |
| `text5` | Required | STRING | "" | Multi-line text | Fifth text input |
| `filter_empty_line` | Required | BOOLEAN | False | True/False | Remove empty lines after processing |
| `filter_comment` | Required | BOOLEAN | False | True/False | Filter `#` line comments and triple-quote blocks (`'''`/`"""`) |
| `separator` | Required | STRING | "\\n" | - | Join separator; supports \\n, \\t, \\r escapes |
| `input` | Optional | STRING | "" | - | Dynamic input value referenced as `{input}` |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Joined result string |
