# String Resolution Qwen Image 3.0 Pro - Qwen Image 3.0 Pro Resolution Selector

**Node Purpose:** `String Resolution Qwen Image 3.0 Pro` outputs the nearest Qwen Image 3.0 Pro resolution tier label (`1k` / `2k`) for each input image based on image area. If no image is connected, it passes through the manually selected label. It is designed to pair with Qwen Image 3.0 Pro.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `1k` / `2k` | Manual Qwen Image 3.0 Pro resolution label used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest resolution tier per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string; batched inputs are joined by newline in batch order. |

## Tiers and target areas

| Tier | 1:1 official size | Target area (pixels) |
| ---- | ----------------- | -------------------- |
| `1k` | `1024x1024` | 1,048,576 |
| `2k` | `2048x2048` | 4,194,304 |

## Features

- Qwen Image 3.0 Pro tiers: `1k` / `2k`, using each tier's official 1:1 size area as the target.
- Area-based matching: compares image area against the tier target areas and picks the nearest in log space.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.

## Typical Usage

- Convert image size into a resolution label before Qwen Image 3.0 Pro nodes that expect a resolution string, and connect it to the API node's `resolution` input.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- The target areas match `QwenImage30Pro.MAX_PIXELS_BY_RESOLUTION` in 1hewOffice (1k=1024², 2k=2048²) item by item; that source belongs to 1hewOffice and its code is not in this repository.
- This node only outputs the string label and does not resize the image.
