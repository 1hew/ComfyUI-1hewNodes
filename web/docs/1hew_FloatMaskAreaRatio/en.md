# Float Mask Area Ratio - Mask area ratio as a 0–1 float

**Node Purpose:** `Float Mask Area Ratio` measures how much of the whole frame a mask covers and returns it as a `0.0–1.0` float, rounded to two decimals by default. Soft mask values are weighted (a `0.5` pixel counts as half a covered pixel), so feathered and anti-aliased masks report their real coverage instead of a full pixel. Use it to turn a mask into a numeric parameter for thresholds, scaling, branching or reporting.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Mask (or mask batch) to measure; its own `H×W` is treated as the whole image, and a batch is averaged across frames. |
| `round` | - | INT | `2` | `0–10` | Decimal places kept on the output ratio; `0` rounds to a whole number (`0.0` or `1.0`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `float` | FLOAT | Covered fraction of the frame, rounded to `round` places (`0.0–1.0`). |

## Features

- Area ratio: the `float` output is `Σ mask / (B × H × W)`, i.e. the mean mask weight.
- Default precision: two decimals, configurable from `0` to `10` via `round`.
- Soft-aware: partial values are weighted, so blur and anti-aliasing keep the same total coverage.
- Batch-aware: a `[B, H, W]` batch returns its average coverage in one value.
- Empty-safe: `None`, empty tensors and non-mask inputs return `0.0`.
- Out-of-range guard: values outside `0–1` are clipped before averaging.

## Typical Usage

- Threshold gating: branch when the subject covers more than a chosen share of the frame.
- Strength scaling: drive denoise or blend strength from the subject's share of the frame.
- Reporting: emit a one-number summary of how much the mask occupies.

## Notes & Tips

- The mask's own resolution is the reference frame, so a full-frame mask at a different scale (e.g. latent-sized) still reports the same ratio.
- A fully black mask returns `0.0`; a fully white mask returns `1.0`.
- For a binary mask the result is simply foreground pixels ÷ total pixels.
- `round` follows the same name and range as `Range Mapping`'s `round` input, keeping decimal control consistent across nodes.
