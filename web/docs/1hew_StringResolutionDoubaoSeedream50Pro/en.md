# String Resolution Doubao Seedream 5.0 Pro - Doubao Seedream 5.0 Pro Resolution Selector

**Node Purpose:** `String Resolution Doubao Seedream 5.0 Pro` outputs the nearest Doubao Seedream 5.0 Pro resolution tier label (`1k` / `1.5k` / `2k`) for each input image based on image area. If no image is connected, it passes through the manually selected label. It is designed to pair with Doubao Seedream 5.0 Pro.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `1k` / `1.5k` / `2k` | Manual Doubao Seedream 5.0 Pro resolution label used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest resolution tier per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string; batched inputs are joined by newline in batch order. |

## Tiers and target areas

| Tier | 1:1 official size | Target area (pixels) |
| ---- | ----------------- | -------------------- |
| `1k` | `1024x1024` | 1,048,576 |
| `1.5k` | `1536x1536` | 2,359,296 |
| `2k` | `2048x2048` | 4,194,304 |

## Features

- Doubao Seedream 5.0 Pro tiers: `1k` / `1.5k` / `2k`, using each tier's official 1:1 size area as the target.
- Area-based matching: compares image area against the tier target areas and picks the nearest in log space.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.

## Typical Usage

- Convert image size into a resolution label before Doubao Seedream 5.0 Pro nodes that expect a resolution string, and connect it to the API node's `resolution` input.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- The target areas match the 1:1 official size of each tier in `DoubaoSeedream50Pro.SIZE_MAP` in 1hewOffice item by item; that source belongs to 1hewOffice and its code is not in this repository.
- This node only outputs the string label and does not resize the image.
