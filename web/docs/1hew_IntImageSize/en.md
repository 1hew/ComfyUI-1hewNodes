# Int Image Size - Width and height integers from image

**Node Purpose:** `Int Image Size` outputs two integers: the image width and height, read directly from the input tensor. This utility node is useful for size-aware pipelines, branching, and parameterization.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B×H×W×C`); spatial dimensions are read. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `width` | INT | Image width in pixels. |
| `height` | INT | Image height in pixels. |

## Features

- Dimension readout: extracts `H` and `W` from image tensors.
- Batch-compatible: supports batched inputs; uses spatial size from tensor shape.
- Lightweight utility: minimal overhead for parameterization and validation.

## Typical Usage

- Control resizing: feed width/height to resizers or pad/crop logic.
- Conditional checks: branch processing based on width/height thresholds.

## Notes & Tips

- Assumes `B×H×W×C` tensor layout; channel dimension is ignored.
- Consistent `H/W` across batch frames yields deterministic outputs.