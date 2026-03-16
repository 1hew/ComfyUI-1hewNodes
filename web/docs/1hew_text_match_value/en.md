# Text Match Value - Text Key-Value Matching

**Node Purpose:** `Text Match Value` matches a specified single-line text against a multi-line text (key-value pairs) and returns the corresponding value. Supports exact match and prefix match, suitable for simple dictionary lookups or conditional routing.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_multiline` | - | STRING | - | - | Multi-line text containing key-value pairs (format like `key: value` or `{key}: {value}`). |
| `text_single` | - | STRING | - | - | The single-line query text to match. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `value` | * (ANY) | The matched value, or an empty string if no match is found. |

## Features

- Key-value parsing: parses `text_multiline` line by line, supporting colon (`:` or `：`) separated key-value pairs.
- Brace unwrapping: automatically removes outer braces `{}` from keys and values, making it convenient to use with wildcards or specific formats.
- Matching strategy:
  1. First tries an exact match (case-insensitive and ignoring leading/trailing spaces).
  2. If no exact match, tries a prefix match (i.e., whether the key starts with the query text).
- Fallback: safely returns an empty string when no match is found.

## Typical Usage

- Prompt mapping: outputting a full prompt paragraph based on a simple tag (e.g., "style_a").
- Parameter routing: selecting corresponding numerical values or configuration strings based on an input string condition.

## Notes & Tips

- The separator between key and value must be an English colon `:` or a Chinese colon `：`.
- The matching process is case-insensitive.