# List Custom String - Custom String List

**Node Purpose:** `List Custom String` parses multi-line text into a list of strings and returns the list alongside its count. It supports dashed-section mode and CSV-style parsing with multiple separators and quote handling.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` | - | Multi-line text to parse into string values. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string_list` | LIST | Parsed string values (list output). |
| `count` | INT | Length of `string_list`. |

## Features

- Dashed-section parsing: dashed-only lines split sections; each section is quote-trimmed and appended as a string.
- CSV-style parsing: supports `,`, `;`, `，`, `；`; trims quotes per item.
- Robust defaults: empty input yields `["default"]`.
- Core logic: dashed detection; dashed parse; CSV parser.

## Typical Usage

- CSV input: `apple, banana, cherry` on one line or multiple lines.
- Section mode: separate values using dashed lines and optional quotes.

## Notes & Tips

- Quotes around items are removed before appending.
- When no valid items are present, the node returns `["default"]`.