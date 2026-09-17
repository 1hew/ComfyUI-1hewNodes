# Image Resize Jimeng - Jimeng Standard Resizer

**Node Purpose:** `Image Resize Jimeng` is designed to resize images and masks to standard resolutions commonly used by Jimeng and other generation models (1k, 2k, 4k, 2.0 pro). It simplifies the workflow by providing preset resolutions and intelligent auto-matching based on input aspect ratios.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k \| 4k)` | `auto` / `auto (1k)` / `auto (2k)` / `auto (4k)` / `auto (1k \| 2k)` / `auto (2k \| 4k)` / `[1k]` / `[2k]` / `[4k]` / `[2.0_pro]` presets | Target resolution preset. `auto` picks the `1k`/`2k`/`4k` tier by input area, then matches the closest aspect ratio within it (excluding `[2.0_pro]`); `auto (1k)` etc. fix the tier. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: determines how the image adapts to the target resolution. |
| `pad_color` | - | STRING | `1.0` | color name / HEX / RGB / `edge` / `extend` / `mirror` / `average` | Padding background color or strategy used when `fit` is set to `pad`. |
| `image` | optional | IMAGE | - | - | Input image batch to be resized. |
| `mask` | optional | MASK | - | - | Input mask batch to be resized. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch. |
| `mask` | MASK | Resized mask batch. |
| `native_image` | IMAGE | Native-resolution copy of the main output; identical content/crop/padding, different resolution. |
| `native_mask` | MASK | Mask aligned with `native_image`. |

## Features

- **Standard Presets**: Built-in support for 1k, 2k, 4k, and 2.0 pro resolution standards, covering common aspect ratios (21:9, 16:9, 3:2, 4:3, 1:1, etc.).
- **Auto Matching**: 
  - `auto`: Picks the `1k`/`2k`/`4k` tier by input area, then matches the closest aspect ratio within it (excluding `[2.0_pro]`).
  - `auto (1k)` / `auto (2k)` / `auto (4k)`: Restrict matching to the selected tier.
  - `auto (1k | 2k)`: Restricts matching to 1k and 2k presets.
  - `auto (2k | 4k)`: Restricts matching to 2k and 4k presets.
- **Fit Modes**:
  - `crop`: Center crops the image to fill the target resolution, preserving aspect ratio.
  - `pad`: Scales the image to fit within the target resolution and pads the empty space, preserving aspect ratio.
  - `stretch`: Stretches the image to exactly match the target resolution, ignoring aspect ratio.
- **Padding Control**: Custom `pad_color` support (e.g., `1.0` for white, `0.0` for black, or specific color codes) ensures flexibility for different model requirements.

## Typical Usage

- **Preparing for Generation**: Select a specific preset like `[2k] 2048x2048 (1:1)` to ensure your input image strictly matches the model's preferred training resolution.
- **Batch Processing**: Use `auto` mode to process a batch of images with varying aspect ratios, automatically resizing each to its nearest standard equivalent.
