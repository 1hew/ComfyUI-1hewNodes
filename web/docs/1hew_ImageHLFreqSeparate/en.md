# Image HL Freq Separate - High/Low Frequency Separation

**Node Purpose:** `Image HL Freq Separate` separates each image into high-frequency and low-frequency components using `rgb`, `hsv`, or `igbi` methods, and returns the separated layers along with a recombined preview.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `method` | - | COMBO | `rgb` | `rgb` / `hsv` / `igbi` | Separation method. |
| `blur_radius` | - | FLOAT | 10.0 | 0.0–100.0 | Gaussian blur radius; internally ensured odd and ≥3.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `high_freq` | IMAGE | High-frequency layer batch. |
| `low_freq` | IMAGE | Low-frequency layer batch. |
| `combine` | IMAGE | Recombined result using the selected method.

## Features

- Odd radius: blur radius is rounded to an odd number ≥3.
- Methods:
- `rgb`: low = Gaussian blur; high = grayscale−blur + 0.5 (clamped), expanded to RGB. Recombine via linear light.
- `hsv`: high from V−blur + 0.5; low from HSV with blurred V; recombine in HSV.
- `igbi`: custom invert+blur mixing for high; low via Gaussian blur; recombine with levels.
- Batch processing: processes each image independently and stacks results.

## Typical Usage

- Detail retouch: extract high for texture work and low for tone work; use `combine` to preview.
- Frequency-aware blending: feed `high_freq` and `low_freq` into `Image HL Freq Combine` for strength-controlled recomposition.

## Notes & Tips

- Inputs are converted to NumPy for OpenCV operations; outputs are float32 in `[0,1]`.
- The `combine` output uses the same method as separation to illustrate intended recomposition.