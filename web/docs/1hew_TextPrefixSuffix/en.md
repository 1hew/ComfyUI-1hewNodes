# Text Prefix Suffix - Per-Item Decoration

**Node Purpose:** `Text Prefix Suffix` decorates each item of the input text (or text list) with a prefix and suffix, then joins items with newline `\n` into a single string. Useful for adding labels, brackets, or markers uniformly per line.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | optional | ANY | - | - | Any text or iterable; non-string iterables are converted to a list for per-item processing |
| `prefix` | - | STRING | `""` | - | Per-item prefix (e.g., `[` or `Scene: `) |
| `suffix` | - | STRING | `""` | - | Per-item suffix (e.g., `]` or `;`) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | String joined with newline after applying prefix/suffix per item |

## Features

- Per-item decoration: applies `prefix` and `suffix` to each element.
- Input normalization:
  - If `text` is a non-string iterable (e.g., generator), it is converted to a list first;
  - If `text` is a single string or scalar, it is wrapped into a single-item list.
- Fixed separator: output is joined using newline `\n`, keeping items on separate lines.

## Typical Usage

- Add brackets per line: `prefix="["`, `suffix="]"`.
- Add unified labels to scene descriptions: `prefix="Scene: "`, `suffix=""`.

## Notes & Tips

- If you need custom separators (e.g., `\n---\n` or other composite forms), use `Text List to String` or `String Join Multi`, which support escape sequences and composite separators.
- For nested lists, inner lists will participate via Python's `str(item)`; flatten nested structures upstream or use nodes that support flattening if needed.