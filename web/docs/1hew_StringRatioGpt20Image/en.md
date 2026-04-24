# String Ratio Gpt20Image - GPT 2.0 Image Ratio Selector

**Node Purpose:** `String Ratio Gpt20Image` outputs the nearest GPT 2.0 image supported aspect ratio string for each input image. If no image is connected, it passes through the manually selected ratio.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `3:2` | `3:2` / `1:1` / `2:3` / `5:4` / `4:5` / `16:9` / `9:16` / `21:9` / `3:4` / `4:3` | Manual GPT 2.0 image ratio preset used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest supported ratio per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Ratio string result; batched inputs are joined by newline in batch order. |

## Features

- GPT 2.0 image ratio set: uses the ratio collection shown in the GPT 2.0 image ratio selector UI.
- Image-driven inference: matches each input image to the nearest preset ratio in log-ratio space.
- Manual fallback: when no image is connected, returns the selected ratio directly.
- Batch-friendly: supports multi-image batches and returns one ratio per image line.

## Typical Usage

- Use before GPT 2.0 image nodes that require an aspect-ratio string.
- Keep a fixed downstream ratio by leaving `image` empty and selecting the target ratio manually.

## Notes & Tips

- This node only outputs the ratio string; it does not resize images.
- Supported ratios are limited to the GPT 2.0 image set and intentionally exclude the extreme Gemini-only presets.
