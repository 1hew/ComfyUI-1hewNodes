# List Custom Float - Custom Float List

**Node Purpose:** `List Custom Float` parses multi-line text into a list of floats and returns the list alongside its count. It supports dashed-section mode and CSV-style parsing with multiple separators and quote handling.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` | - | Multi-line text to parse into float values. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `float_list` | LIST | Parsed float values (list output). |
| `count` | INT | Length of `float_list`. |

## Features

- Dashed-section parsing: lines consisting solely of `-` split sections; each section is stripped of quotes and parsed as a float.
- CSV-style parsing: supports `,`, `;`, `，`, `；` separators; trims quotes per item.
- Robust defaults: empty input yields `[0.0]`; invalid tokens are ignored.
- Core logic: dashed detection; dashed parse; CSV parser.

## Typical Usage

- CSV input: `1.0, 0.5, 2.75` on one line or multiple lines.
- Section mode: separate values using dashed lines to declare individual entries.

## Notes & Tips

- Quotes around numbers (e.g., `'0.5'`) are removed before parsing.
- When no valid numbers are present, the node falls back to `[0.0]`.