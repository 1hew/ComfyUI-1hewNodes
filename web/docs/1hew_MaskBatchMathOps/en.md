# Mask Batch Math Ops - Reduce masks by OR/AND

**Node Purpose:** `Mask Batch Math Ops` reduces a mask batch across the batch dimension using logical-maximum (`or`) or logical-minimum (`and`). Outputs a single aggregated mask as a one-frame batch.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch; supports `B×H×W` or `B×H×W×1`. |
| `operation` | - | COMBO | `or` | `or` / `and` | Reduction operator across batch. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Single-frame aggregated mask (`B=1`), clamped to `[0,1]`. |

## Features

- OR reduction: per-pixel maximum across frames.
- AND reduction: per-pixel minimum across frames.
- Channel handling: auto-squeezes masks with a trailing singleton channel and restores shape via `unsqueeze(0)`.
- Chunked processing: reduces in chunks of `512` frames to limit memory usage.
- Output normalization: converts to `float32`, clamps `[0,1]`, and preserves device.

## Typical Usage

- Combine multiple segmentation results into a single consensus mask via `or`.
- Enforce strict overlap regions using `and`.

## Notes & Tips

- For batches of size `≤1`, the input mask is returned unchanged.