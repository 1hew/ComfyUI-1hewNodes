# String Filter - Text Cleaner

**Node Purpose:** `String Filter` performs placeholder substitution, comment filtering, and optional empty-line removal, outputting a clean string. It supports single-line comments `#` and triple-quote blocks `"""` / `'''`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | - | STRING | `""` | - | Input text, multiline supported; literal `\n` is converted to newline. |
| `filter_empty_line` | - | BOOLEAN | `false` | - | Remove empty lines. |
| `filter_comment` | - | BOOLEAN | `false` | - | Filter comments (`#` and triple-quote blocks). |
| `input` | - | STRING | `""` | - | Value to replace `{input}` placeholders. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Filtered text string. |

## Features

- Placeholder substitution: replaces `{input}` before processing.
- Comment filtering:
  - Single-line: `#` and trailing content are removed.
  - Multi-line: supports `""" ... """` and `''' ... '''` blocks; remains active until closed.
- Empty-line control: optional removal; trailing whitespace is stripped (right trim).
- Newline normalization: literal `\n` becomes real newline.

## Typical Usage

- Clean text before joining; use with `String Join Multi` or `Text List to String`.
- Inject external values using `{input}` in template text for unified substitution.

## Notes & Tips

- Triple-quote blocks must be properly paired; same-line closure is handled.
- If you only want to drop empty lines, set `filter_comment=false`.