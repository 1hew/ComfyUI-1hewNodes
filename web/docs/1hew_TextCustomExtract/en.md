# Text Custom Extract - Structured Text Extraction

**Node Purpose:** `Text Custom Extract` extracts values for a target `key` from JSON-like text. It supports robust input cleaning, multiple parse fallbacks, enhanced key synonyms, optional precision matching, and label-based filtering.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `json_data` | - | STRING | `` | - | JSON or JSON-like text; code fences and leading/trailing non-JSON content are cleaned. |
| `key` | - | STRING | `zh` | - | Target key to extract (supports synonyms). |
| `precision_match` | - | COMBO | `disabled` | `disabled` / `enabled` | Exact-key matching when `enabled`; otherwise case-insensitive with synonyms. |
| `label_filter` | - | STRING | `` | - | Optional comma-separated labels to filter items (supports Chinese/English). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Extracted value(s); lists of numbers are joined with commas; list items from arrays are joined by newlines. |

## Features

- Input cleaning: removes code fences and trims to the first balanced JSON block.
- Multi-strategy parsing: tries `json.loads`, falls back to `ast.literal_eval`, quote normalization, and regex key-value extraction.
- Enhanced key synonyms: supports variations for `bbox`, `label`, `confidence`, `x`, `y`, `width`, `height`, `zh`, `en`.
- Label filtering: optional filter on a `label` field for lists or objects.
- Precision matching: exact-key when enabled; otherwise case-insensitive and synonym-aware.
- Value formatting: numeric lists joined by commas; other lists stringified.
- Core logic: cleaning; parsing; synonyms; lookup; format; filters; main execution.

## Typical Usage

- Simple extraction: set `key=zh` to extract Chinese text from objects or arrays.
- Synonym use: set `key=bbox` to match `bbox`, `box`, `bounding_box`, etc.
- Label filtering: set `label_filter=person,cat` to retain items whose labels contain `person` or `cat`.
- Exact matching: set `precision_match=enabled` to require the exact provided key.

## Notes & Tips

- The node accepts JSON fenced by Markdown code blocks and trims extraneous text around JSON.
- When input is an array of objects, matched values are joined with newlines in output.