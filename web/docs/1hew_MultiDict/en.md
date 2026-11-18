# Multi Dict - Merge dicts and add one key/value

**Node purpose:** `Multi Dict` merges dynamic `dict_1..N` inputs in port order and then adds or overwrites one key from `key` / `value`. Later keys win, and `key` / `value` are applied last so they always override the merged result. `value_type` controls how `value` is converted. It outputs a `dict` object and a `json` string.

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `value_type` | - | COMBO | `auto` | `auto` / `string` / `int` / `float` / `boolean` / `dict` / `list` / `json` / `null` | Conversion applied to `value` only; see below. |
| `dict_1`, `dict_2`… | optional | * | - | - | Dicts merged in port order. Only dicts are accepted; connect each dict of a list to a separate `dict_N` socket. |
| `key` | optional | * | - | - | Key added or overwritten last. Strings and scalars are accepted (numbers/booleans are stringified). Unconnected or empty means merge only. |
| `value` | optional | * | - | - | Value added or overwritten last, after `value_type` conversion. Ignored when `key` is empty; an unconnected value becomes `null`. |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `dict` | DICT | Merged dict for the next `Multi Dict` `dict_1` or `Save Text`. |
| `json` | STRING | Compact JSON text for `Save Text`, LLMs and other text nodes. |

## `value_type` conversions

| Value | Behavior | Example |
| ----- | -------- | ------- |
| `auto` | Keep the native value | `"30"` → `"30"`; `{...}` → object |
| `string` | Convert to text; dict/list become compact JSON text | `30` → `"30"`; `{"a":1}` → `"{\"a\":1}"` |
| `int` | Convert to integer; numeric strings are parsed and fractions are truncated | `"30"` → `30`; `3.7` → `3`; `"3.7"` → `3` |
| `float` | Convert to float | `"3.5"` → `3.5`; `2` → `2.0` |
| `boolean` | Convert to boolean; `true` / `1` / `yes` / `on` / non-zero numbers are true | `"true"` → `true`; `"0"` → `false` |
| `dict` | Parse JSON text and require an object | `'{"a":1}'` → object; `[1,2]` raises |
| `list` | Parse JSON text and require an array | `'[1,2]'` → array; `{"a":1}` raises |
| `json` | Parse a string as JSON; objects→dict, arrays→list, scalars map to their JSON types | `'{"a":1}'` → object |
| `null` | Force `null` | any value → `null` |

- `value_type` affects `value` only; `key` is still converted to a string/scalar key.
- An unconnected `value` always becomes `null`, regardless of `value_type`; a connected value that cannot be converted raises an error.
- `json` is strict: plain text such as `Alice` raises; use `auto` or `string` to keep normal text.

## Order and overwrite

1. `dict_1` → `dict_2` → `dict_3`… merge in numeric socket order;
2. a later duplicate key overwrites the earlier value and keeps its first position;
3. `key` / `value` are applied last (after `value_type`), so they override every merged dict.

Example:

```text
dict_1 = {a: 1, b: 1}
dict_2 = {a: 2}
key = a, value = 9
result = {a: 9, b: 1}
```

## Notes

- Only dicts are accepted: a string, number or list on `dict_N` raises an error. To merge several dicts from a list, connect each one to its own `dict_N` socket.
- `key` accepts strings, integers, floats and booleans; dicts and lists raise an error.
- When `key` is unconnected or blank, the node only merges dicts; a connected `value` is ignored with a console message.
- When `key` is valid but `value` is unconnected, the node outputs `{key: null}`.
- String values are never wrapped: in `auto` a connected string stays a string instead of becoming `["string"]`. For an array, connect a real list value such as a `Multi List` `list` output.
- Only JSON-safe values are accepted: set, bytes, tensors, NaN and Infinity raise path-aware errors instead of being silently stringified.

## Typical use

- Single member: `key=<String name>`, `value=<String Alice>` gives `{"name": "Alice"}`.
- JSON text to dict: `value=<String '{"a":1}'>` with `value_type=json` gives `{"k": {"a": 1}}`.
- Numeric text to number: `value=<String "30">` with `value_type=int` gives `{"k": 30}`.
- Merge: `dict_1=<Multi Dict A>`, `dict_2=<Multi Dict B>` merges both in order.
- Chaining: `D2(dict_1=D1, key="b", value=2)` gives `{...D1, "b": 2}`.
- Save: connect `dict` or `json` to `Save Text` with `format=json`.

## Notes and tips

- `key` / `value` are socket-only inputs; literals come from `String` / `Int` / `Float` / `Boolean` nodes.
- The front end keeps `key` / `value` after every dynamic `dict_N` socket so the visual order matches merge-then-append.
- JSON keys must be strings; an empty key means merge only.
