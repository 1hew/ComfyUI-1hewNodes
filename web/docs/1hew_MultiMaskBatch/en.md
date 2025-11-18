# Multi Mask Batch - Align Masks to Reference Size

**Node Purpose:** `Multi Mask Batch` aligns multiple masks to the size of the first mask using `crop`, `pad`, or `stretch`. Pads with a configurable value and concatenates all aligned masks into one batch.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `fit` | - | COMBO | `pad` | `crop` / `pad` / `stretch` | Alignment mode. |
| `pad_color` | - | FLOAT | 0.0 | 0.0–1.0 | Padding value for `pad` mode. |
| `mask_1` | - | MASK | - | - | First mask; defines reference size. |
| `mask_2…mask_N` | optional | MASK | - | - | Additional masks recognized by numeric suffix ordering.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Batch of aligned masks; clamped to `[0,1]` float.

## Features

- Reference size: uses height/width of `mask_1` as target.
- Fit modes:
- `stretch`: bilinear resize directly to target.
- `crop`: scale to cover target, then center-crop.
- `pad`: scale to fit, then pad with `pad_color`.
- Ordering: collects `mask_*` inputs by numeric suffix to preserve sequence.

## Typical Usage

-- Unify mask batches: standardize sizes for logical operations or compositing.
-- Use `pad_color` to distinguish padding as background in subsequent processing.
-- Cover vs fit: choose `crop` for full coverage, `pad` to preserve entire masks.

## Notes & Tips

- Padding applies a constant value across padded regions; set `pad_color` according to downstream semantics.
- All outputs are single-channel masks aligned and stacked along batch dimension.