# Data Path Extract - Read values by path

**Node Purpose:** `Data Path Extract` reads values out of the connected `data` input with a path expression. The data can be JSON text, JSON inside a Markdown fence, a Python literal, relaxed JSON (single quotes, trailing commas, True/False/None, unquoted keys), plain text, or an already-parsed dict / list. The default path `[*]` splits the whole payload into its values; paths such as `[0].content`, `[-1].bbox[0]`, `data.items[0].zh`, `values[0].ids` and `keys[*]` pick the matching content.

The node only collects values: `value` is always a list with one entry per match. Rendering that list into text is the next node's job, so `List Text`, `Text List to String` and `Save Text` can each format it the way they need.

## Inputs

| Name | Type | Default | Range | Description |
| ---- | ---- | ------- | ----- | ----------- |
| `data` | * (required link) | - | - | The payload, connected: JSON text, fenced JSON, a Python literal, or an already-parsed dict / list |
| `data_path` | STRING | `[*]` | - | JSON path; `[*]` means every value, an empty path means the whole payload. Named `data_path` rather than `path` so it does not read as a file path |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `value` | ANY (list) | Always a list, one entry per match, `[]` when nothing matches or parsing fails. Values keep their type: lists stay lists, numbers stay numbers |
| `count` | INT | Number of matches; the amount itself when the path ends with `.count` / `.length` / `[#]`; 0 when the path misses or the input is empty. After a run the port label shows the number (e.g. `3 count`) |

## Path Syntax

| Form | Meaning |
| ---- | ------- |
| `[0]` / `[-1]` | Index like Python: `0` is the first item, `-1` the last; on an object it takes the value at that position (insertion order), so an unknown key name is still addressable |
| `.content` / `content` | Dict key; the leading dot is optional |
| `[0]["content"]` / `[0].'content'` | Quoted key, for names containing dots or spaces; quoting is also what separates the key `"0"` from position `0` |
| `keys[*]` / `keys[0]` | An object's key names: all of them / the nth one; a list has no key names, so `keys` finds nothing there (use `values[*].id` for ids) |
| `values[*]` / `values[0]` | An object's values: all of them / the nth one — the default path is `[*]` (the same as `values[*]`); on a list it means every item / the nth item |
| `values[*].content` | Wildcard: apply `content` to every value |
| `$.data.items[0].zh` | Optional JSONPath-style root `$` |
| `values[*].content.count` / `[0].bbox.length` / `[0].bbox[#]` | Count: the number of matches, or the length of the matched list / dict / string |
| empty path | The whole payload itself |

`keys` / `values` also accept the singular spellings `key` / `value`; a real field with the same name wins (see the notes).

Full-width `【】`, `“”` and `‘’` are normalised, so a Chinese IME typing `【0】.“content”` still works.

## Features

- Values only, no formatting: `value` hands over the raw list, so the same result can feed `List Text` for a prefix, `Save Text` for a file and `List Math Expression` for arithmetic at the same time.
- Layered parsing: strict JSON, Python literal, relaxed JSON (trailing commas, True/False/None), a loose parser for unquoted keys, then a key/value scrape for truncated payloads.
- Fence cleaning: Markdown code fences and prose around the JSON block are stripped.
- Double decoding: a payload that is itself a JSON string, such as `"{\"a\": 1}"`, is decoded once more.
- Exact keys: path keys are case-sensitive and are not synonym-matched; use `Text Custom Extract` when synonyms are wanted.
- Key / value selectors: `keys[*]` and `values[*]` take all key names or all values of an object and can be followed by `[n]`, `values[*]` or `.field` — write `values[0].ids` when the key name is unknown, and `keys[0]` to see which key was hit; on a list `values[*]` walks the items (a list has no key names, so `keys` only applies to objects).
- Counting: the `count` output is always the number of matched items; a trailing `.count` / `.length` / `.size` / `[#]` returns that number as the value itself, and the `count` output then carries it too.
- Front-end value: after a run the `count` port label is prefixed with the number (e.g. `3 count`), matching the `count` value; reconnecting an input restores the plain `count` label until the next run.

## Typical Usage

- First detection text: `json_path=[0].content` gives `value = ["犀牛"]`.
- Every content: `json_path=values[*].content` gives `["犀牛","犀牛","犀牛"]`.
- Every bbox: `json_path=values[*].bbox` gives `[[184,410,845,716],[508,679,1000,953],[690,264,992,588]]`.
- The amount: wire the `count` output, or use the path `values[*].content.count`.
- Read a reference image whose key name varies (`reference_image_1` or `image_1`): `json_path=values[0].ids` takes the first image's ids; with several images `values[0]` / `values[-1]` take the first / last, `values[*].ids` takes all of them, and `keys[0]` shows which key was hit.
- Add one to every index and write it back: `value` → `List Math Expression` (`a + 1`) → `Data Path Replace`.
- Render the list as text: feed `List Text` (prefix / suffix per item) or `Text List to String` (join with a separator).
- Save it as a JSON file: feed `Save Text` with `format` set to `json`.

## Notes & Tips

- Blank `text` returns an empty list with `count` 0 and never raises.
- A missing path also returns an empty list; let downstream handle the empty list.
- If the parser raises on an unexpected payload, the result is an empty list and `count` 0 (the error is printed to the console), so downstream nodes never receive a type the port does not declare.
- Plain text has no items to walk, so `values[*]` finds nothing; leave `data_path` empty to pass plain text through unchanged.
- A single match still arrives as a list (`[value]`), not a bare value; take the first item downstream, or use `List Text`.
- A real `count` key is not shadowed: if the object has that key its value wins; write `["count"]` to force literal key access. `keys` / `values` (singular too) follow the same rule — a payload that really has a `values` field returns it, and only `["values"]` is guaranteed to be the literal key.
- An index on an object reads the value at that position, but a real key wins: in `{"0": …}` `[0]` returns the key `"0"`. For a pure position write `values[0]`: the selector produces the value list first, so the index can no longer be intercepted by a key name.
- A list has no key names: `keys[*]` misses there (the result is an empty list); use `values[*].id` for `box_001`… and `values[*].keys[*]` for each item's field names.
- Object positions follow insertion order (`json.loads` keeps the text order); if the producer starts sorting keys, `[0]` can point at a different item, so use `keys[*]` to check or read the key name directly when the order must be stable.
- The numeric label comes from the `web/js/output_label.js` extension; without it only the label is missing, the returned values are unaffected.
