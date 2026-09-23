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

## Tiers available per node / model

This node ships every tier preset, but the Jimeng `resolution` options are not uniform across nodes, so pick a tier the paired node actually supports:

| Jimeng node | Available tiers |
| ----------- | --------------- |
| `Jimeng Image T2I` | `1k` / `2k` / `4k` |
| `Jimeng Image Multi Edit` | `2k` / `4k` |
| `Jimeng Image 4.0` / `4.1` / `4.5` / `4.6` / `4.7` / `5.0 Lite` | `2k` / `4k` |
| `Jimeng Image 3.0` / `ControlNet` / `IP` | `1k` / `2k` |
| `Jimeng Image Single Edit` / `Style` / `Subject` | `1k` / `2k` |

- Model `2.0_pro` uses a separate 1024-base size table (`2.0_pro_ratios`), which maps to this node's `[2.0_pro]` presets and is not part of the `1k`/`2k`/`4k` tiers; that is why `auto` never selects `[2.0_pro]`.
- When paired with `String Resolution Jimeng`, keep the label it outputs within the tiers listed above.
