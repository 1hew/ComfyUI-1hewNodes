# Mask List to Batch - Merge list into padded batch

**Node Purpose:** `Mask List to Batch` merges a list of masks into a single batch. Normalizes shape and pads each mask to the maximum height/width using zeros so concatenation is possible.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask_list` | list | MASK/LIST | - | - | Input masks as a list or single mask. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask_batch` | MASK | Combined batch (`B×H×W`) after zero padding to common size. |

## Features

- Flexible input: accepts a single tensor or a list/tuple; non-tensor items are ignored.
- Shape normalization: converts `H×W` to `1×H×W` for batching.
- Size reconciliation: pads each mask to the maximum `H` and `W` across inputs using zeros.
- Fallback: returns an empty batch of shape `(0, 64, 64)` when no valid masks are found; returns the single mask unchanged when only one is present.

## Typical Usage

- Combine variable-sized masks into a batch aligned for model inputs.

## Notes & Tips

- Padding uses constant value `0` and preserves device/dtype before concatenation.