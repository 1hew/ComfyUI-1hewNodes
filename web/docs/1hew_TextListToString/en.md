# Text List to String - Merge Text List

**Node Purpose:** `Text List to String` takes a text list (including nested lists), applies per-item `prefix`/`suffix`, and joins them with a `separator` into a single string. Escape sequences and composite separators like `\n---\n` are supported.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_list` | optional | ANY | - | - | Text list or nested lists; when absent, result is an empty string. |
| `prefix` | - | STRING | `""` | - | Per-item prefix. |
| `suffix` | - | STRING | `""` | - | Per-item suffix. |
| `separator` | - | STRING | `"\n"` | supports `\n` / `\t` / `\r` / `\\` combos | Separator between items; composite separators like `\n---\n` are supported. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Joined string result. |

## Features

- Nested flattening: expands sub-lists and preserves order.
- Per-item decoration: applies `prefix`/`suffix` to every item.
- Separator escaping: supports `\n`, `\t`, `\r`, `\\` and composite strings (e.g., `\n---\n`).
- Robust inputs: `None` or empty lists yield an empty string; blank items are preserved and processed consistently.

## Typical Usage

- Merge `string_list` from `List Custom String` into paragraphs (e.g., `\n` or `\n---\n`).
- Produce a comma-separated label string: set separator to `, ` and use `prefix`/`suffix` to wrap items (e.g., quotes).

## Notes & Tips

- For nested lists, the node flattens in order; if you need to filter empty lines or comments, use `String Filter` upstream or pre-process inputs.