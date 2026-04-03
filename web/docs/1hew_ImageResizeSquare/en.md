# Image Resize Square - Square Size Adapter

**Node Purpose:** `Image Resize Square` provides general-purpose square presets with `256`, `512`, `1024`, `2048`, and `4096`, plus `auto` and `auto (0.5k | 1k)` matching modes. It is intended for workflows that need to normalize images and masks into square outputs.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto` | `auto` / `auto (0.5k \| 1k)` / `[0.25k] 256x256 (1:1)` / `[0.5k] 512x512 (1:1)` / `[1k] 1024x1024 (1:1)` / `[2k] 2048x2048 (1:1)` / `[4k] 4096x4096 (1:1)` | Square target size selector; `auto (0.5k \| 1k)` matches within the `512` and `1024` tiers first. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, or stretch. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB/color name/`edge`/`average`/`extend`/`mirror` | Background fill strategy for `pad` mode. |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch. |
| `mask` | MASK | Output mask batch aligned to output size. |

## Features

- Square-only presets: keeps only `1:1` targets for square workflows.
- Auto selection: `auto` chooses the nearest square target across all five tiers by input area; `auto (0.5k | 1k)` limits matching to `512` and `1024`.
- Same fit behavior: `crop` / `pad` / `stretch` stay consistent with the existing resize nodes, including synchronized mask transforms.
- Flexible inputs: supports image-only, mask-only, and no-input fallback.

## Typical Usage

- Medium/small square preprocessing: `preset_size=auto (0.5k | 1k)`.
- Force a fixed square output: directly choose `256`, `512`, `1024`, `2048`, or `4096`.
- Preserve the full subject: use `fit=pad` with a suitable `pad_color`.

## Notes & Tips

- For faster preprocessing, prefer `512`, `1024`, or `auto (0.5k | 1k)`.
- Use `auto` when you want the node to adapt across the full square range from `256` to `4096`.
