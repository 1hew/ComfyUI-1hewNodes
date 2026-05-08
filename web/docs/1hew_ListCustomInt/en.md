# List Custom Int - Flexible integer list syntax

**Node Purpose:** `List Custom Int` parses multiline text into an integer list and returns the list alongside its count. It supports single values, lists, forward/reverse ranges, ranged strides, mixed Chinese/English punctuation, spaces, and mixed bracket styles. Single float values are coerced with `int(float(...))`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` | multiline text | Text to parse using the flexible range/list syntax. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int_list` | LIST | Parsed integer values (list output). |
| `count` | INT | Length of `int_list`. |

## Features

- Loose parsing: accepts `,`, `，`, `:`, `：`, `[]`, `【】`, `()`, `（）`, including mixed usage.
- Single values and lists: supports `1`, `2.9`, `1,2,3`, `1，2，3`.
- Ranges and stride: supports `0-10`, `10-0`, `0-10:2`, `[0,10]`, `[0,10)`, `【10，0】：2`.
- Reverse ranges: expressions like `10-0` and `[10,0]:2` expand in reverse order.
- Order and duplicates preserved: values are emitted in written order and duplicates are kept.
- Robust defaults: empty input yields `[0]`; invalid fragments are ignored.

## Typical Usage

- Single values and lists: `1, 2, 3`
- Continuous range: `[0,10]`
- Strided range: `[0,20):2`
- Reverse range: `【10，0】：2`
- Mixed input:
  ```text
  1，2，3
  [10,20)
  【30，20】：2
  7.9
  ```

## Notes & Tips

- `:` / `：` means stride only in this node.
- Single float values are coerced with `int(float(...))`, for example `7.9 -> 7`.
- Range endpoints and stride are parsed with integer semantics.
- When no valid integers are present, the node falls back to `[0]`.