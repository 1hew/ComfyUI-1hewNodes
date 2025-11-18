# List Custom Int - Custom Integer List

**Node Purpose:** `List Custom Int` parses multi-line text into a list of integers and returns the list alongside its count. It supports dashed-section mode and CSV-style parsing with multi-language separators and quote handling, including decimal-to-integer coercion.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` | - | Multi-line text to parse into integer values. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int_list` | LIST | Parsed integer values (list output). |
| `count` | INT | Length of `int_list`. |

## Features

- Dashed-section parsing: each dashed block yields one integer value.
- CSV-style parsing: supports `,`, `;`, `，`, `；`; trims quotes per item.
- Decimal coercion: items like `"3.9"` are converted via `int(float(...))`.
- Robust defaults: empty input yields `[0]`; invalid tokens are ignored.
- Core logic: dashed detection; dashed parse; CSV parser; decimal coercion.

## Typical Usage

- CSV input: `1, 2, 3` on one line or multiple lines.
- Mixed numeric strings: `"7"`, `"8.2"`, `"9"` become `7, 8, 9`.
- Section mode: separate values using dashed lines.

## Notes & Tips

- Quotes around numbers are removed before parsing.
- When no valid integers are present, the node falls back to `[0]`.