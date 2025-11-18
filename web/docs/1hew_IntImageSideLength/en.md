# Int Image Side Length - Image width/height/longest/shortest

**Node Purpose:** `Int Image Side Length` returns a single integer derived from the input image size: longest side, shortest side, width, or height. Designed for size-aware pipelines to set or validate downstream parameters.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B×H×W×C`); only spatial dimensions are used. |
| `mode` | - | COMBO | `longest` | `longest` / `shortest` / `width` / `height` | Select which side length to output. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Selected side length in pixels. |

## Features

- Dimension readout: extracts `H` and `W` from images and computes per `mode`.
- Batch-compatible: supports batched tensors where all frames share the same spatial size.
- Efficient utility: minimal overhead to drive size-related parameters.

## Typical Usage

- Control resizing: feed as the target side length for universal resizers or pad/crop logic.
- Validate inputs: assert width/height ranges to gate downstream processing.

## Notes & Tips

- Assumes image tensors follow `B×H×W×C` layout; channels are not used.
- For batched inputs, consistent `H/W` across frames yields deterministic outputs.