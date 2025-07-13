# Text Format

**Node Function:** The `Text Format` node is used to format any input data into text strings with customizable prefix, suffix, and separator, supporting wildcard input types for flexible text processing and formatting.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `any_text` | Required | * | - | - | Any input data to be formatted (supports wildcard input types) |
| `prefix` | - | STRING | "" | - | Text prefix to add before each item |
| `suffix` | - | STRING | "" | - | Text suffix to add after each item |
| `separator` | - | STRING | "\\n" | - | Separator to join multiple items (use \\n for newline) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `text` | STRING | Formatted text string |