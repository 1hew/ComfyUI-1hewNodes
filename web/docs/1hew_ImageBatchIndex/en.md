# Image Batch Index - Select images by index text

**Node Purpose:** `Image Batch Index` extracts specific images from an image batch using multiline index text. It supports single indices, lists, forward/reverse ranges, ranged strides, and mixed Chinese/English punctuation. Out-of-range indices are skipped automatically.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `index` | - | STRING | `0` | multiline text | Index expressions with flexible punctuation, spaces, and mixed brackets. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image batch extracted from the index rules. |

## Supported Syntax

- Single index: `0`, `2`, `-1`
- List: `0,2,6`, `0，2，6`
- Legacy range: `0-10`, `10-0`, `0-10:2`
- Bracket range: `[0,10]`, `[0,10)`, `(0,10]`, `(0,10)`
- Mixed brackets: `【0，10】`, `【0，10)`, `(10，0】`
- Range stride: `[0,20):2`, `【10，0】：2`

## Features

- Loose parsing: accepts `,`, `，`, `:`, `：`, `[]`, `【】`, `()`, `（）`, and mixed usage.
- Multiline input: each line can contain one or more expressions.
- Negative indices: values like `-1` refer to images from the end.
- Reverse ranges: expressions like `10-0` and `[10,0]:2` expand in reverse order.
- Order preserved: output follows the written order of expressions.
- Duplicates preserved: repeated indices produce repeated images in the output.
- Out-of-range skip: parsed indices outside the batch are ignored automatically.

## Typical Usage

- Pick specific images: `0, 2, 6, -1`
- Take a continuous span: `[0,10]`
- Sample by stride: `[0,20):2`
- Reverse selection: `【10，0】：2`
- Mixed input:
  ```text
  0，2，6
  [10,20)
  【30，20】：2
  -1
  ```

## Notes & Tips

- `:` / `：` means stride only in this node.
- `0-10` is treated as inclusive on both ends.
- If all parsed indices are invalid or out of range, the output is an empty batch.
