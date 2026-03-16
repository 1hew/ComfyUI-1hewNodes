# text_match_rownum - Multi-line Text Row Matching

**Node Purpose:** `text_match_rownum` searches a target string within multi-line text and returns the first matched row number (1-based). It returns `0` when no match is found.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_multiline` | - | STRING | - | - | Multi-line input text (split by line breaks). |
| `text_single` | - | STRING | - | - | Single-line target text to match. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `rownum` | INT | First matched row index (1-based); returns `0` if not found. |

## Features

- Matching uses trimmed whole-line equality (`strip()` on both sides).
- Returns only the first match, suitable for routing and indexing scenarios.
- Safe fallback to `0` on empty input or runtime exceptions.

## Typical Usage

- Text routing: match one keyword against candidate lines and feed `rownum` into switch-like logic nodes.
- Config selection: map row index to different parameter groups or processing branches.

## Notes & Tips

- Row numbers are 1-based, not 0-based.
- For contains/regex-style matching, preprocess text upstream, then use this node for deterministic row lookup.
