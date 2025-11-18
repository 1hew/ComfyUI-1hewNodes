# Text Prefix Suffix - Format and Join

**Node Purpose:** `Text Prefix Suffix` formats input values by applying `prefix` and `suffix` to each item and joins them with a separator. It flexibly accepts single values, lists/tuples, and other iterables.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `any_text` | - | ANY | - | - | Single value, list/tuple, or iterable; items are converted to strings. |
| `prefix` | - | STRING | `` | - | Text prepended to each item. |
| `suffix` | - | STRING | `` | - | Text appended to each item. |
| `separator` | - | STRING | `\\n` | - | Item joiner; `\\n` yields newline; other values are used as provided. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Joined text after per-item formatting. |

## Features

- Iterable handling: expands non-string iterables into a list; otherwise wraps single value.
- Separator rule: uses `\n` only when `separator` equals `\\n`; otherwise uses `str(separator)`.
- Per-item formatting: applies `prefix`/`suffix` and converts items to string.

## Typical Usage

- Format lists: feed a list or generator and set `prefix`/`suffix` for uniform wrapping.
- Multi-line output: keep `separator=\\n` for one item per line.
- Custom joiners: set `separator=", "` or other delimiters for compact strings.

## Notes & Tips

- When escape decoding for `\t`, `\r`, or `\\` is desired, pair with `Text Join by Text List` or `Text Join Multi` to decode additional sequences.
- Accepts general iterables, enabling direct formatting from generators or sets.