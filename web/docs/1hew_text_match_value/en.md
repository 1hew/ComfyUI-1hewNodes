# Text Match Value - Text Key-Value Matching

**Node Purpose:** `Text Match Value` matches a specified single-line text against a multi-line text (key-value pairs) and returns the corresponding value. It supports exact match and prefix match, and automatically converts simple boolean and numeric text into typed values for routing.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_multiline` | - | STRING | - | - | Multi-line text containing key-value pairs (format like `key: value` or `{key}: {value}`). |
| `text_single` | - | STRING | - | - | The single-line query text to match. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `value` | * (ANY) | The matched value; automatically recognizes `true/false`, integers, and floats, or returns an empty string when no match is found. |

## Features

- Key-value parsing: parses `text_multiline` line by line, supporting colon (`:` or `：`) separated key-value pairs.
- Brace unwrapping: automatically removes outer braces `{}` from keys and values, making it convenient to use with wildcards or specific formats.
- Automatic type coercion: `50` becomes an integer, `3.14` becomes a float, `true/false` become booleans, and all other values remain strings.
- Matching strategy:
  1. First tries an exact match (case-insensitive and ignoring leading/trailing spaces).
  2. If no exact match, tries a prefix match (i.e., whether the key starts with the query text).
- Fallback: safely returns an empty string when no match is found.

## Typical Usage

- Prompt mapping: outputting a full prompt paragraph based on a simple tag (e.g., "style_a").
- Parameter routing: selecting corresponding numerical values or configuration strings based on an input string condition.
- Numeric mapping: entries like `24:50`, `54:40`, and `70:30` can now feed numeric node inputs directly.

## Notes & Tips

- The separator between key and value must be an English colon `:` or a Chinese colon `：`.
- The matching process is case-insensitive.
- To avoid accidental coercion, integer text with leading zeros (for example `001`) stays as a string; use normal numeric literals when you want numeric output.