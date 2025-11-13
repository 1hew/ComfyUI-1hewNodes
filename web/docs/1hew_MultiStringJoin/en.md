# Multi String Join

**Node Function:** The `Multi String Join` node concatenates dynamic `string_X` inputs into a single string, with per-input comment filtering, empty-line control, `{input}` placeholder substitution, and customizable separators.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `string_1` | Optional | STRING | - | - | First dynamic text input; supports additional `string_2`, `string_3`, ... |
| `filter_empty_line` | Required | BOOLEAN | False | True/False | Remove empty lines after processing each text |
| `filter_comment` | Required | BOOLEAN | False | True/False | Filter line comments (`#`) and triple-quote blocks (`'''`/`"""`) |
| `separator` | Required | STRING | "\\n" | - | Concatenation separator; supports \\n, \\t, \\r escapes |
| `input` | Optional | STRING | "" | - | Dynamic input value referenced as `{input}` |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Joined result string |
