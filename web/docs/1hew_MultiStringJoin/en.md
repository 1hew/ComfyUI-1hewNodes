# Multi String Join - Ordered Join with Filtering and Placeholder

**Node Purpose:** `Multi String Join` joins multiple string inputs in numeric order using a configurable separator, with options to filter empty lines and comments, and to substitute `{input}` placeholders.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `filter_empty_line` | - | BOOLEAN | False | - | Remove empty lines after processing. |
| `filter_comment` | - | BOOLEAN | False | - | Remove comments and triple-quoted blocks. |
| `separator` | - | STRING | `\n` | - | Join separator; supports escapes `\n`, `\t`, `\r`. |
| `input` | - | STRING | `` | - | Value to substitute into `{input}` placeholders. |
| `string_1` | - | STRING | - | - | First string.
| `string_2…string_N` | optional | STRING | - | - | Additional strings recognized by numeric suffix ordering.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Joined and filtered result.

## Features

- Ordered collection: gathers `string_*` inputs by numeric suffix.
- Placeholder substitution: replaces `{input}` inside each string and final result.
- Comment filtering: removes `#` comments and respects `'''`/`"""` block quotes with stateful parsing.
- Empty-line filtering: optional removal after trimming.
- Separator decoding: interprets escape sequences for clean joins.

## Typical Usage

- Prompt assembly: combine prompt fragments with placeholders for dynamic content.
- Config templating: join sections while stripping comments and blank lines.
- Ordered merging: ensure `string_1..N` join in intended sequence.

## Notes & Tips

- Block-quote handling ensures multi-line quoted content is preserved when filtering comments.
- Provide `separator="\n\n"` for paragraph breaks, or `", "` for comma-separated lists.