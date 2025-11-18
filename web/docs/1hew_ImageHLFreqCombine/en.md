# Image HL Freq Combine - Recombine High/Low Frequency

**Node Purpose:** `Image HL Freq Combine` recombines a high-frequency image with a low-frequency image using one of three methods: `rgb`, `hsv`, or `igbi`. It provides separate strength controls for the high and low layers and robust batch alignment.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `high_freq` | - | IMAGE | - | - | High-frequency layer batch. |
| `low_freq` | - | IMAGE | - | - | Low-frequency layer batch. |
| `method` | - | COMBO | `rgb` | `rgb` / `hsv` / `igbi` | Recombination method. |
| `high_strength` | - | FLOAT | 1.0 | 0.0–2.0 | Strength multiplier for the high layer; `rgb/hsv` are centered at 0.5. |
| `low_strength` | - | FLOAT | 1.0 | 0.0–2.0 | Strength multiplier for the low layer. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Recombined image batch.

## Features

- Strength shaping: for `rgb/hsv`, the high layer is offset around mid-gray `(high-0.5)*high_strength+0.5`; `igbi` scales directly.
- Methods:
- `rgb`: linear light recombination `2*high + low - 1`.
- `hsv`: linear light on V channel while preserving H/S from `low`.
- `igbi`: weighted mix `0.65*high + 0.35*low` followed by levels adjustment.
- Batch alignment: repeats smaller batches to match the larger one.

## Typical Usage

- Sharpening finish: separate high/low elsewhere, then recombine with `rgb` and adjust `high_strength`.
- Tonal integration: use `hsv` to affect brightness only while preserving hue and saturation from the low layer.
- Stylized blend: choose `igbi` for a more contrast-managed result via built-in levels.

## Notes & Tips

- Input layers should be aligned in size and content; the node will handle batch count alignment but not spatial misalignment.
- Values are clamped to `[0,1]` before recombination and output is clamped afterwards.