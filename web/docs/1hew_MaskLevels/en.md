# Mask Levels - Mask Levels Adjustment

**Node Purpose:** `Mask Levels` applies a levels adjustment to a grayscale mask, remapping the 0–255 range via `black_point` / `white_point`: values at or below `black_point` are pushed to 0, values at or above `white_point` are pulled to 255, and the in-between range is linearly interpolated. When the two points coincide it degrades to a binary threshold.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask; accepts 2D, `[B,H,W]`, or `[B,H,W,C]`. |
| `black_point` | - | INT | 0 | 0–255 (step 1) | Black point; pixels at or below this gray level (normalized) map to 0. |
| `white_point` | - | INT | 255 | 0–255 (step 1) | White point; pixels at or above this gray level map to 255 (i.e. 1.0). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Level-adjusted mask in the 0–1 range. |

## Features

- Linear stretch: the `(black_point, white_point)` interval is linearly interpolated and clamped to `[0, 1]`.
- Inversion: when `black_point > white_point`, the slope is negative, naturally producing an inverted mask.
- Binary fallback: when `black_point == white_point`, it becomes a binary threshold (`>=` value → 1, otherwise 0).
- Shape compatibility: 2D, `[B,H,W]`, and `[B,H,W,C]` inputs are handled (4D uses the first channel of the last dimension).
- Safe empty input: invalid shapes return an empty `[0,64,64]` mask instead of raising.

## Typical Usage

- Clean up masks: compress noise and boost contrast with black/white points to turn a soft mask into a cleaner range.
- Invert a mask: set `black_point > white_point` to flip the mask directly.

## Notes & Tips

- The defaults (`black_point=0`, `white_point=255`) are effectively passthrough, mapping the mask to 0–1 unchanged.
- With `black_point=3` and `white_point=235`: 0–3 → 0, 235–255 → 1, with a linear transition in between.
