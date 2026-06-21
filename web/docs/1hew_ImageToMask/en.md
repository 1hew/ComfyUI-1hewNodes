# Image to Mask - Convert Image Luminance to Mask

**Node Purpose:** `Image to Mask` converts an optional image input into a grayscale `MASK` using RGB luminance. When no image is connected, it outputs a 64x64 black mask.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | optional | - | Image to convert into a mask. If omitted, the node outputs a 64x64 black mask |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Grayscale mask generated from image luminance, or a 64x64 black mask when no image is connected |

## Features

- Optional image input for workflows that need a fallback mask.
- Uses RGB luminance weights: `0.299 * R + 0.587 * G + 0.114 * B`.
- Preserves batch size and image dimensions when an image is connected.
- Outputs a batch of one 64x64 black mask when the image input is empty.

## Typical Usage

- Convert a grayscale or RGB guide image into a mask.
- Provide a safe black-mask fallback for optional image branches.
- Prepare image-derived masks for downstream mask operation nodes.
