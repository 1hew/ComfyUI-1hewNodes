# List Custom Float - Flexible float list syntax

**Node Purpose:** `List Custom Float` parses multiline text into a float list and returns the list alongside its count. It supports single values, lists, forward/reverse ranges, ranged strides, mixed Chinese/English punctuation, spaces, and mixed bracket styles. Float ranges are cleaned to avoid common accumulation noise.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` | multiline text | Text to parse using the flexible range/list syntax. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `float_list` | LIST | Parsed float values (list output). |
| `count` | INT | Length of `float_list`. |

## Features

- Loose parsing: accepts `,`, `，`, `:`, `：`, `[]`, `【】`, `()`, `（）`, including mixed usage.
- Single values and lists: supports `1.0`, `.5`, `1.0, 2.5, 3`.
- Ranges and stride: supports `0-1`, `1-0`, `0-1:0.25`, `[0,1]`, `[0,1)`, `【1，0】：0.2`.
- Reverse ranges: expressions like `1-0` and `[1,0]:0.2` expand in reverse order.
- Precision cleanup: output values are normalized to avoid artifacts like `0.30000000000000004`.
- Order and duplicates preserved: values are emitted in written order and duplicates are kept.
- Robust defaults: empty input yields `[0.0]`; invalid fragments are ignored.

## Typical Usage

- Single values and lists: `1.0, 0.5, 2.75`
- Continuous range: `[0,1]`
- Strided range: `[0,1):0.25`
- Reverse range: `【1，0】：0.2`
- Mixed input:
  ```text
  1.0，0.5，2.75
  [0,1):0.25
  【1，0】：0.2
  .125
  ```

## Notes & Tips

- `:` / `：` means stride only in this node.
- Both range endpoints and stride support floating-point values.
- When no valid numbers are present, the node falls back to `[0.0]`.