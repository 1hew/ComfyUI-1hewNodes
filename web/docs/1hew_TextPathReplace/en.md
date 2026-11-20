# Text Path Replace - Write values back at a path

**Node purpose:** `Text Path Replace` writes values into JSON-like text at the places a path matches. It is the write half of `Text Path Extract`: the same path that reads a value also says where a new value goes. The value is typed into `value` and converted by `value_type`; this node only places it.

Like `Text Path Extract` the node does not render. The edited document leaves through the `value` port as a one-item list, and formatting is the next node's job — `Save Text` to write a file, `List Text` to decorate, or any JSON dump downstream.

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `text` | - | STRING | `` | multiline | The document to edit: JSON, fenced JSON or a Python literal |
| `json_path` | - | STRING | `values[*]` | - | Path to write to; the same syntax as `Text Path Extract` |
| `value` | - | STRING | `` | - | The value to write, typed in directly and converted by `value_type`. A single value is broadcast to every match |
| `value_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `dict` / `list` / `json` / `null` | Converts `value`, the same meaning as `Multi Dict`. `auto` keeps text as text, so pick `int` / `float` for numbers |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `value` | ANY (list) | The edited document as a one-item list, so a root-level JSON array is not split by ComfyUI downstream |
| `count` | INT | Number of locations written; 0 when the path matches nothing |

## Replacement rules

| `value` items | Matches | Result |
| ------------- | ------- | ------ |
| 1 | N | The single value is written to all N locations |
| N | N | One-to-one, in match order |
| any other count | N | Error: the counts do not match, nothing is written |

## Paths that cannot be written

- `keys[*]` / `key[0]`: key names are read-only here, so a replacement is rejected.
- `count` / `length` / `size` / `[#]` operators: read-only, rejected.
- A bare `values` with no index or `[*]`: the whole group has no single position, so it is rejected.

## Typical usage

- Set every content to the same value: `path = [*].content` with one `value` — it broadcasts to each match.
- Replace every content with one string: `path = [*].content`, a single `value` item is broadcast.
- Replace a nested coordinate: `path = [1].bbox[0]`, `value` `50` with `value_type` set to `int`.
- Save the result: wire `value` into `Save Text` with `format` set to `json`.

## Notes

- The input text is never changed in place; the node works on a copy and returns the edited document.
- If the path matches nothing, `value` holds the original document unchanged and `count` is 0.
- A mismatch or an invalid path raises an error instead of silently writing somewhere else.
- A value that cannot be represented as JSON (for example a Python set) is rejected with an error.
- `value_type=auto` keeps the upstream type as it is: a string `"2"` stays a string. Choose `int` / `float` when the JSON needs a number.
