# Text Path Extract - Read values by path

**Node Purpose:** `Text Path Extract` reads a value out of the `text` widget with a path expression. The text can be JSON, JSON inside a Markdown fence, a Python literal, relaxed JSON (single quotes, trailing commas, True/False/None, unquoted keys) or plain text. The default path `values[*]` splits the whole payload into its values; paths such as `[0].content`, `[-1].bbox[0]`, `data.items[0].zh`, `values[0].ids` and `keys[*]` pick the matching content.

## Inputs

| Name | Type | Default | Range | Description |
| ---- | ---- | ------- | ----- | ----------- |
| `text` | STRING | `` | multiline | The payload: JSON / JSON-like / plain text |
| `path` | STRING | `values[*]` | - | Path expression; `values[*]` means every value, an empty path means the whole payload |
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
| `[0]` / `[-1]` | Index like Python: `0` is the first item, `-1` the last; on an object it takes the value at that position (insertion order), so an unknown key name is still addressable |
| `.content` / `content` | Dict key; the leading dot is optional |
| `[0]["content"]` / `[0].'content'` | Quoted key, for names containing dots or spaces; quoting is also what separates the key `"0"` from position `0` |
| `keys[*]` / `keys[0]` | An object's key names: all of them / the nth one; a list has no key names, so `keys` finds nothing there (use `values[*].id` for ids) |
| `values[*]` / `values[0]` | An object's values: all of them / the nth one — the default path is `values[*]`; on a list it means every item / the nth item |
| `values[*].content` | Wildcard: apply `content` to every value |
| `$.data.items[0].zh` | Optional JSONPath-style root `$` |
| `values[*].content.count` / `[0].bbox.length` / `[0].bbox[#]` | Count: the number of matches, or the length of the matched list / dict / string |
| empty path | The whole payload itself |

`keys` / `values` also accept the singular spellings `key` / `value`; a real field with the same name wins (see the notes).

Full-width `【】`, `“”` and `‘’` are normalised, so a Chinese IME typing `【0】.“content”` still works.

## Features

- Layered parsing: strict JSON, Python literal, relaxed JSON (trailing commas, True/False/None), a loose parser for unquoted keys, then a key/value scrape for truncated payloads.
- Fence cleaning: Markdown code fences and prose around the JSON block are stripped.
- Double decoding: a payload that is itself a JSON string, such as `"{\"a\": 1}"`, is decoded once more.
- Exact keys: path keys are case-sensitive and are not synonym-matched; use `Text Custom Extract` when synonyms are wanted.
- Key / value selectors: `keys[*]` and `values[*]` take all key names or all values of an object and can be followed by `[n]`, `values[*]` or `.field` — write `values[0].ids` when the key name is unknown, and `keys[0]` to see which key was hit; on a list `values[*]` walks the items (a list has no key names, so `keys` only applies to objects).
- Counting: the `count` output is always the number of matched items; a trailing `.count` / `.length` / `.size` / `[#]` returns that number as the value itself, and the `count` output then carries it too.
- Front-end value: after a run the `count` port label is prefixed with the number (e.g. `3 count`), matching the `count` value; reconnecting an input restores the plain `count` label until the next run.
- Whole-set JSON: with several matches and `output_format=json` the output is one JSON array (`values[*].bbox` gives `[[184,410,845,716],...]`) instead of one value per line; the other formats join with `separator`.

## Typical Usage

- See every value: `path=values[*]` prints one value per line.
- Take the first detection text: `path=[0].content`.
- Take the last box for a downstream node: `path=[-1].bbox` with `output_format=json` gives `[690,264,992,588]`.
- Get every content as a list: `path=values[*].content` leaves the list on `value`; with `output_format=json` the `text` output is `["犀牛","犀牛","犀牛"]`.
- Get every bbox as a list: `path=values[*].bbox` with `output_format=json` gives `[[184,410,845,716],[508,679,1000,953],[690,264,992,588]]`.
- Get the amount: wire the `count` output, or use the path `values[*].content.count`.
- Read a reference image whose key name varies (`reference_image_1` or `image_1`): `path=values[0].ids` takes the first image's ids; with several images `values[0]` / `values[-1]` take the first / last, `values[*].ids` takes all of them, and `keys[0]` shows which key was hit.
- Get a comma-separated coordinate string: `path=[0].bbox` with `output_format=csv` gives `184, 410, 845, 716`.
- Keep the structure: wire the `value` output instead of the string output.

## Notes & Tips

- Blank `text` returns `fallback` directly and never raises.
- A missing path also returns `fallback`; leave it empty to pass an empty value downstream.
- If the parser raises on an unexpected payload, `text` and `value` both return `fallback` and `count` is 0 (the error is printed to the console), so downstream nodes never receive a type the port does not declare.
- Plain text has no items to walk, so `values[*]` finds nothing; leave `path` empty to pass plain text through unchanged.
- With `auto` strings stay raw while lists / dicts become compact JSON; choose `json` when strings must keep their quotes.
- A real `count` key is not shadowed: if the object has that key its value wins; write `["count"]` to force literal key access. `keys` / `values` (singular too) follow the same rule — a payload that really has a `values` field returns it, and only `["values"]` is guaranteed to be the literal key.
- An index on an object reads the value at that position, but a real key wins: in `{"0": …}` `[0]` returns the key `"0"`. For a pure position write `values[0]`: the selector produces the value list first, so the index can no longer be intercepted by a key name.
- A list has no key names: `keys[*]` misses there (the fallback comes back); use `values[*].id` for `box_001`… and `values[*].keys[*]` for each item's field names.
- Object positions follow insertion order (`json.loads` keeps the text order); if the producer starts sorting keys, `[0]` can point at a different item, so use `keys[*]` to check or read the key name directly when the order must be stable.
- The numeric label comes from the `web/js/output_label.js` extension; without it only the label is missing, the returned values are unaffected.
