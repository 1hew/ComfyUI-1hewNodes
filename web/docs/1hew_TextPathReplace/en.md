# Text Path Replace - Write values back at a path

**Node purpose:** `Text Path Replace` writes values into JSON-like text at the places a path matches. It is the write half of `Text Path Extract`: the same path that reads a value also says where a new value goes. Computation stays upstream (`List Math Expression`, `Math Expression`, `Multi Dict`, string nodes); this node only places the replacement values.

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `text` | - | STRING | `` | multiline | JSON, fenced JSON or a Python literal to edit |
| `path` | - | STRING | `values[*]` | - | Path to write to; the same syntax as `Text Path Extract` |
| `replace` | - | * (list) | - | - | Values to write. A list is applied in order; a single value is broadcast to every match |
| `value_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `dict` / `list` / `json` / `null` | Converts `replace` items, the same meaning as `Multi Dict` |
| `output_format` | - | COMBO | `pretty` | `pretty` / `compact` | How `text` is rendered |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `text` | STRING | The edited document, pretty-printed or compact |
| `value` | * list | The edited document wrapped in a one-item list, so a root-level JSON array is not split downstream |
| `count` | INT | Number of locations written; 0 when the path matches nothing |

## Replacement rules

| `replace` items | Matches | Result |
| --------------- | ------- | ------ |
| 1 | N | The single value is written to all N locations |
| N | N | One-to-one, in match order |
| any other count | N | Error: the counts do not match, nothing is written |

## Paths that cannot be written

- `keys[*]` / `key[0]`: key names are read-only here, so a replacement is rejected.
- `count` / `length` / `size` / `[#]` operators: read-only, rejected.
- A bare `values` with no index or `[*]`: the whole group has no single position, so it is rejected.

## Typical usage

- Add one to every index: `Text Path Extract` (`layers[*].index`) → `List Math Expression` (`a + 1`) → `Text Path Replace` (`path = layers[*].index`, `replace` from the math output).
- Replace every content with one string: `path = [*].content`, a single `replace` item is broadcast.
- Replace a nested coordinate: `path = [1].bbox[0]`, `replace = [50]`.

## Notes

- The input text is never changed in place; the node works on a copy and returns the edited document.
- If the path matches nothing, `text` is the original input and `count` is 0.
- A mismatch or an invalid path raises an error instead of silently writing somewhere else.
- A value that cannot be represented as JSON (for example a Python set) is rejected with an error.
