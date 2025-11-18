# Text Join by Text List - Join Formatted Items

**Node Purpose:** `Text Join by Text List` joins items from a list into a single text, applying `prefix` and `suffix` to each item, with a customizable separator that supports common escape sequences.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_list` | - | ANY | - | - | Input list or single value; single values are wrapped into a list. |
| `prefix` | - | STRING | `` | - | Text prepended to each item. |
| `suffix` | - | STRING | `` | - | Text appended to each item. |
| `separator` | - | STRING | `\\n` | - | Item joiner; supports `\\n`, `\\t`, `\\r`, `\\\\`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Joined text from formatted items. |

## Features

- Escape-aware separator: decodes `\\n`→`\n`, `\\t`→`\t`, `\\r`→`\r`, `\\\\`→`\\`.
- Flexible input: accepts list/tuple or wraps single input into a one-item list.
- Per-item formatting: applies `prefix`/`suffix` and converts values to string.

## Typical Usage

- Comma-joined lines: set `separator=", "` to produce a single comma-separated string.
- Quoted items: set `prefix="\""`, `suffix="\""` to wrap each item with quotes.
- Multi-line output: keep `separator=\\n` for line-per-item formatting.

## Notes & Tips

- For advanced filtering of comments or empty lines prior to joining, combine with `Text Filter` or `Text Join Multi`.
- Input items are converted via `str(...)`, enabling numbers or mixed types to join seamlessly.