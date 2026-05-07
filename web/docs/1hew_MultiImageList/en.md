# Multi Image List - Dynamic Image List Builder

**Node Purpose:** `Multi Image List` collects images from dynamic `image_X` inputs and outputs an `image_list`. Inputs are processed in numeric order; each `image_X` can receive a normal image batch or an image list, and all items are split into single-image list entries.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_1` | optional | IMAGE / IMAGE_LIST | - | - | First image input; accepts a batch or list. |
| `image_2...image_N` | optional | IMAGE / IMAGE_LIST | - | - | Additional dynamic image inputs; each accepts a batch or list. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_list` | IMAGE_LIST | Image list ordered by input index. |

## Features

- Dynamic inputs: add more ports such as `image_2`, `image_3`, and beyond.
- Stable ordering: collects inputs by numeric suffix in ascending order.
- Batch/list splitting: each batch or list input is recursively split into single `[1,H,W,C]` image items.
- Empty input skipping: unconnected inputs, empty tensors, and non-image values are ignored.

## Typical Usage

- Convert several independent image ports into an `image_list` for downstream list-based nodes.
- Pair with `Image List to Batch` when the list needs to be converted back into a batch.
