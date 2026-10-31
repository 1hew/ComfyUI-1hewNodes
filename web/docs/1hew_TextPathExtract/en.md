# Text Path Extract - Read values by path

**Node Purpose:** `Text Path Extract` reads a value out of the `text` widget with a path expression. The text can be JSON, JSON inside a Markdown fence, a Python literal, relaxed JSON (single quotes, trailing commas, True/False/None, unquoted keys) or plain text. The default path `[*]` splits the whole payload into its items; paths such as `[0].content`, `[-1].bbox[0]`, `data.items[0].zh` and `[*].content` pick the matching content.

## Inputs

| Name | Type | Default | Range | Description |
| ---- | ---- | ------- | ----- | ----------- |
| `text` | STRING | `` | multiline | The payload: JSON / JSON-like / plain text |
| `path` | STRING | `[*]` | - | Path expression; `[*]` means every item, an empty path means the whole payload |
| `output_format` | COMBO | `auto` | `auto` / `json` / `text` / `csv` | Affects the `text` output only; `auto` means smart rendering (see below) |
| `separator` | STRING | `\n` | supports `\n` / `\t` / `\r` | Joins multiple matches (every format except `json`) |
| `fallback` | STRING | `` | - | Returned when the path misses or the input is empty |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Rendered text; with several matches `json` emits one array, the other formats join with `separator` |
| `value` | ANY | The raw value, never stringified: lists stay lists, numbers stay numbers; a list when several matches |
| `count` | INT | Number of matches; the amount itself when the path ends with `.count` / `.length` / `[#]`; 0 when the path misses or the input is empty. After a run the port label shows the number (e.g. `3 count`) |

## output_format

Using `path=[0].bbox` (a list of coordinates), `output_format` only changes `text`, never `value`:

| Value | Meaning | `text` for that example |
| ----- | ------- | ---- |
| `auto` | Smart: strings stay raw, lists / dicts become compact JSON, numbers and bools become text | `[184,410,845,716]` |
| `json` | Always JSON; strings keep their double quotes | `[184,410,845,716]` |
| `text` | Python `str()` style: lists with spaces, strings without quotes | `[184, 410, 845, 716]` |
| `csv` | Lists joined with a comma and a space, no brackets | `184, 410, 845, 716` |

## Path Syntax

| Form | Meaning |
| ---- | ------- |
| `[0]` / `[-1]` | List index, negatives allowed |
| `.content` / `content` | Dict key; the leading dot is optional |
| `[0]["content"]` / `[0].'content'` | Quoted key, for names containing dots or spaces |
| `$.data.items[0].zh` | Optional JSONPath-style root `$` |
| `[*]` / `[*].content` / `data.*.content` | Wildcard: every item (the default path is `[*]`) |
| `[*].content.count` / `[0].bbox.length` / `[0].bbox[#]` | Count: the number of matches, or the length of the matched list / dict / string |
| empty path | The whole payload itself |

Full-width `【】`, `“”` and `‘’` are normalised, so a Chinese IME typing `【0】.“content”` still works.

## Features

- Layered parsing: strict JSON, Python literal, relaxed JSON (trailing commas, True/False/None), a loose parser for unquoted keys, then a key/value scrape for truncated payloads.
- Fence cleaning: Markdown code fences and prose around the JSON block are stripped.
- Double decoding: a payload that is itself a JSON string, such as `"{\"a\": 1}"`, is decoded once more.
- Exact keys: path keys are case-sensitive and are not synonym-matched; use `Text Custom Extract` when synonyms are wanted.
- Wildcard collection: `[*]` walks list items or dict values and applies the remaining path to each.
- Counting: the `count` output is always the number of matched items; a trailing `.count` / `.length` / `.size` / `[#]` returns that number as the value itself, and the `count` output then carries it too.
- Front-end value: after a run the `count` port label is prefixed with the number (e.g. `3 count`), matching the `count` value; reconnecting an input restores the plain `count` label until the next run.
- Whole-set JSON: with several matches and `output_format=json` the output is one JSON array (`[*].bbox` gives `[[184,410,845,716],...]`) instead of one value per line; the other formats join with `separator`.

## Typical Usage

- See every item: `path=[*]` prints one element per line.
- Take the first detection text: `path=[0].content`.
- Take the last box for a downstream node: `path=[-1].bbox` with `output_format=json` gives `[690,264,992,588]`.
- Get every content as a list: `path=[*].content` leaves the list on `value`; with `output_format=json` the `text` output is `["犀牛","犀牛","犀牛"]`.
- Get every bbox as a list: `path=[*].bbox` with `output_format=json` gives `[[184,410,845,716],[508,679,1000,953],[690,264,992,588]]`.
- Get the amount: wire the `count` output, or use the path `[*].content.count`.
- Get a comma-separated coordinate string: `path=[0].bbox` with `output_format=csv` gives `184, 410, 845, 716`.
- Keep the structure: wire the `value` output instead of the string output.

## Notes & Tips

- Blank `text` returns `fallback` directly and never raises.
- A missing path also returns `fallback`; leave it empty to pass an empty value downstream.
- If the parser raises on an unexpected payload, `text` and `value` both return `fallback` and `count` is 0 (the error is printed to the console), so downstream nodes never receive a type the port does not declare.
- Plain text has no items to walk, so `[*]` finds nothing; leave `path` empty to pass plain text through unchanged.
- With `auto` strings stay raw while lists / dicts become compact JSON; choose `json` when strings must keep their quotes.
- A real `count` key is not shadowed: if the object has that key its value wins; write `["count"]` to force literal key access.
- The numeric label comes from the `web/js/output_label.js` extension; without it only the label is missing, the returned values are unaffected.
