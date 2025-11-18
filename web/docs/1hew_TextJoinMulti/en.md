# Text Join Multi - Join Multiple Texts with Filtering

**Node Purpose:** `Text Join Multi` combines up to five multi-line texts, with optional comment and empty-line filtering, and supports `{input}` placeholder substitution and escape-aware separators.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text1` | - | STRING | `` | - | First multi-line text. |
| `text2` | - | STRING | `` | - | Second multi-line text. |
| `text3` | - | STRING | `` | - | Third multi-line text. |
| `text4` | - | STRING | `` | - | Fourth multi-line text. |
| `text5` | - | STRING | `` | - | Fifth multi-line text. |
| `filter_empty_line` | - | BOOLEAN | `False` | - | Remove empty lines after processing. |
| `filter_comment` | - | BOOLEAN | `False` | - | Enable comment filtering for `#` and triple quotes. |
| `separator` | - | STRING | `\\n` | - | Joiner that decodes `\\n`, `\\t`, `\\r`, `\\\\`. |
| `input` | - | STRING | `` | - | Placeholder value for `{input}` substitution. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Joined result after formatting and filtering. |

## Features

- Escape-aware separator: decodes `\\n`→`\n`, `\\t`→`\t`, `\\r`→`\r`, `\\\\`→`\\`.
- Placeholder substitution: replaces `{input}` within each text before filtering.
- Comment filtering: supports `#` line comments and `"""`/`'''` multi-line blocks.
- Flexible empty-line handling: include or skip via `filter_empty_line`.
- Input selection: only non-empty parsed sections participate in joining.

## Typical Usage

- Compose prompts: split prompt parts across `text1..text5`, inject variables via `{input}`, and set filters to keep the result clean.
- Controlled spacing: tune `separator` for commas, newlines, or tabs depending on downstream needs.

## Notes & Tips

- Filtering occurs after `{input}` substitution, ensuring placeholders are evaluated before comment removal.
- Apply consistent filter settings across texts to keep output uniform.