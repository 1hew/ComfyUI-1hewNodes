# Multi Mask Batch

**Node Function:** The `Multi Mask Batch` node builds a mask batch from dynamic `mask_X` inputs, unifying sizes based on the first mask and applying `pad`, `crop`, or `stretch` modes with bilinear resampling.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `mask_1` | Optional | MASK | - | - | First dynamic mask input; supports additional `mask_2`, `mask_3`, ... |
| `fit` | Required | COMBO[STRING] | pad | crop, pad, stretch | Size unification mode for masks |
| `pad_color` | Required | FLOAT | 0.0 | 0.0–1.0 (step 0.01) | Padding value (grayscale) used during `pad` mode |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `masks` | MASK | Unified mask batch with values clamped to `[0, 1]` |
