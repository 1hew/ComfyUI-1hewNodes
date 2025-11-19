# String Join Multi - Join Multiple Text Blocks

**Node Purpose:** `String Join Multi` accepts up to 5 text blocks, supports `{input}` substitution, comment filtering, and empty-line control, then joins them using a separator. Separator supports escape sequences and composite forms like `\n---\n`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text1` | - | STRING | `""` | - | Text block 1, multiline supported. |
| `text2` | - | STRING | `""` | - | Text block 2, multiline supported. |
| `text3` | - | STRING | `""` | - | Text block 3, multiline supported. |
| `text4` | - | STRING | `""` | - | Text block 4, multiline supported. |
| `text5` | - | STRING | `""` | - | Text block 5, multiline supported. |
| `filter_empty_line` | - | BOOLEAN | `false` | - | Remove empty lines. |
| `filter_comment` | - | BOOLEAN | `false` | - | Filter comments (`#` and triple-quote blocks). |
| `separator` | - | STRING | `"\n"` | supports `\n` / `\t` / `\r` / `\\` combos | Join separator; composite like `\n---\n` works. |
| `input` | - | STRING | `""` | - | String to replace `{input}` placeholders. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Joined result string. |

## Features

- Per-block processing: placeholder substitution and comment/empty-line filtering per block; empty results are skipped.
- Separator escaping: supports `\n`, `\t`, `\r`, `\\` and composite strings (e.g., `\n---\n`).
- Template injection: consistent `{input}` substitution across blocks.

## Typical Usage

- Consolidate multiple prompt sections into a single output with visual separators (`separator=\n---\n`).
- Clean blocks (remove comments and empty lines) before joining for a tidy context.

## Notes & Tips

- Only non-empty processed blocks participate in the final join; to preserve empty blocks, set `filter_empty_line=false`.
- Triple-quote blocks must be properly closed; for advanced filtering, combine with `String Filter`.