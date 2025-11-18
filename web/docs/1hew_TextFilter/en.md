# Text Filter - Comment and Empty-Line Filtering

**Node Purpose:** `Text Filter` processes multi-line text with configurable comment and empty-line filtering. It supports single-line `#` comments and triple-quoted multi-line blocks, and converts literal `\\n` into actual newlines.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | - | STRING | `` | - | Input multi-line text; literal `\\n` becomes a newline. |
| `filter_empty_line` | - | BOOLEAN | `False` | - | Remove empty lines after processing. |
| `filter_comment` | - | BOOLEAN | `False` | - | Enable comment filtering for `#` and triple quotes. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Filtered text result. |

## Features

- Escape handling: converts `\\n` to `\n` before filtering.
- Comment filtering: supports `#` line comments and `"""`/`'''` multi-line blocks.
- Whitespace trimming: applies `rstrip` to processed lines outside comment blocks.
- Flexible empty-line handling: include or skip based on `filter_empty_line`.
- Empty input handling: returns an empty string for blank input.

## Typical Usage

- Clean a prompt: set `filter_comment=True` and `filter_empty_line=True` to produce compact, comment-free text.
- Preserve structure: set `filter_empty_line=False` when spacing is meaningful while still removing comments.

## Notes & Tips

- Triple-quoted blocks are treated as comments and excluded when comment filtering is enabled.
- Apply this node before downstream concatenation to keep inputs concise and consistent.