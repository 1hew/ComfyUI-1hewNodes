# Mask Repeat

**Node Purpose:** The `Mask Repeat` node is used to repeat the input mask (supports batch) for a specified number of times, and supports color inversion. It is suitable for scenarios where a single or a group of masks need to be expanded into a larger batch.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | Required | MASK | - | - | Input mask batch. |
| `count` | - | INT | 1 | 1-4096 | Number of repetitions. The entire input Batch is treated as a unit for repetition. |
| `invert` | - | BOOLEAN | False | True/False | Whether to invert mask colors (swap black and white). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | The repeated (and optionally inverted) mask batch. |

## Features

- **Batch Repeat**:
    - If input Mask Batch is `[A, B]` and `count` is 2, the output will be `[A, B, A, B]`.
    - The entire input Batch is treated as a whole for duplication.

- **Invert Function**:
    - When `invert` is `True`, the output mask colors are inverted (0 becomes 1, 1 becomes 0).
    - This operation is performed logically along with repetition.

## Typical Usage

- **Generate Long Sequence Masks**: Expand a single looping mask to the number of frames required for video length.
- **Invert and Expand**: Simultaneously complete mask inversion and batch copying, simplifying the number of nodes in the workflow.
