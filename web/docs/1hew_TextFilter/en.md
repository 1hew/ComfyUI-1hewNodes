# Text Filter

**Node Function:** The `Text Filter` node parses multi-line text and filters comment content, supporting single-line `#` comments and triple-quoted blocks, with optional empty-line removal and escape handling.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `text` | Required | STRING | "" | Multi-line text | Text to parse and filter |
| `filter_empty_line` | Required | BOOLEAN | False | True/False | Remove empty lines after filtering |
| `filter_comment` | Required | BOOLEAN | False | True/False | Enable comment filtering (`#`, `'''`, `"""`) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Filtered text string |

## Features

- Comment filtering: Handles `#` single-line comments and triple-quoted blocks; skips lines that become empty due to filtering when enabled.
- Empty-line control: When enabled, removes non-comment empty lines to produce compact output.
- Escape handling: Translates `\\n` into newline before filtering for accurate line parsing.

