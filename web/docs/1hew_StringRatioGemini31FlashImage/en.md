# String Ratio Gemini31FlashImage - Gemini 3.1 Flash Ratio Selector

**Node Purpose:** `String Ratio Gemini31FlashImage` outputs the nearest Gemini 3.1 Flash supported aspect ratio string for each input image. If no image is connected, it passes through the manually selected ratio.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1:1` | `1:1` / `1:4` / `1:8` / `2:3` / `3:2` / `3:4` / `4:1` / `4:3` / `4:5` / `5:4` / `8:1` / `9:16` / `16:9` / `21:9` | Manual Gemini 3.1 Flash ratio preset used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest supported ratio per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Ratio string result; batched inputs are joined by newline in batch order. |

## Features

- Gemini 3.1 Flash ratio set: uses the same supported ratio family as `Image Resize Gemini31FlashImage`.
- Image-driven inference: matches each input image to the nearest preset ratio in log-ratio space.
- Manual fallback: when no image is connected, returns the selected ratio directly.
- Batch-friendly: supports multi-image batches and returns one ratio per image line.

## Typical Usage

- Use before Gemini 3.1 Flash image nodes that require an aspect-ratio string.
- Pair with `Image Resize Gemini31FlashImage` to keep resize presets and ratio strings aligned.
- Leave `image` empty when you want to force a fixed ratio string downstream.

## Notes & Tips

- This node only outputs the ratio string; it does not resize images.
- The output includes extreme Gemini 3.1 Flash ratios such as `1:8`, `4:1`, and `8:1`.
