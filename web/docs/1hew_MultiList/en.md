# Multi List - Combine elements into an array

**Node purpose:** `Multi List` combines dynamic `item_1..N` inputs into one JSON array in port order. `extend` matches Python `list.extend` and splices list/tuple inputs one level; `append` matches `list.append` and keeps each input as one element, with lists staying nested.

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `mode` | - | COMBO | `extend` | `extend` / `append` | `extend` splices list/tuple inputs; `append` keeps them nested. |
| `item_1`, `item_2`… | optional | * | - | - | Elements combined in numeric socket order; more connect through dynamic sockets. |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `list` | * | Combined array for another `Multi List` or a `Multi Dict` value. |
| `json` | STRING | Compact JSON array text. |

## `mode` semantics

| Inputs `"a"`, `[1, 2]`, `{"b": 3}` | Result |
| ----------------------------------------- | ------ |
| `extend` | `["a", 1, 2, {"b": 3}]` |
| `append` | `["a", [1, 2], {"b": 3}]` |

## Notes

- `extend` only splices `list` / `tuple`: strings are not split into characters and dicts are not split into keys (in JSON they are values, not element groups).
- `append` always adds one element; a list stays a nested array.
- Splitting is one level deep; nested lists stay as elements. Chain more nodes for deeper flattening.
- Scalars stay scalars: a single string, number or dict is one element and is not wrapped in an array.
- Unconnected or `None` sockets are skipped; all-empty produces `[]`.
- Values: set, bytes, tensors, NaN and Infinity raise path-aware errors during serialization.

## Typical use

- Combine scalars, dicts and lists into one array.
- Build object arrays from several `Multi Dict` outputs.
- Expand a single upstream list (`extend`) or keep it as one nested element (`append`).

## Notes and tips

- `item_N` takes elements; to merge dicts use `Multi Dict` instead of feeding a `Multi Dict` `dict` into `item_N` and expecting a merge.
- The `list` output is a single array value, not `OUTPUT_IS_LIST`; feeding it to a `Multi Dict` value keeps it as an array.
