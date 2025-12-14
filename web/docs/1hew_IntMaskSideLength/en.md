# Int Mask Side Length - Mask width/height/longest/shortest

**Node Purpose:** `Int Mask Side Length` returns a single integer based on the input mask size: longest side, shortest side, width, or height. Useful for driving downstream parameters (e.g., target side length) in size-aware workflows.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch (`B×H×W`); only its spatial size is used. |
| `mode` | - | COMBO | `shortest` | `longest` / `shortest` / `width` / `height` | Select which side length to output. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Selected side length in pixels. |

## Features

- Size extraction: reads `H` and `W` from mask and computes per `mode`.
- Batch-friendly: works with mask batches; dimensions are taken from tensor shape.
- Zero overhead: minimal computation, ideal for parameter control.

## Typical Usage

- Drive resizing: feed into nodes that require a target side length (e.g., longest side). 
- Conditional flows: branch by width/height values to select processing paths.

## Notes & Tips

- Expects standard mask shape `B×H×W` with numeric values; only dimensions matter here.
- When masks are batched, all frames should share the same spatial size for deterministic outputs.

