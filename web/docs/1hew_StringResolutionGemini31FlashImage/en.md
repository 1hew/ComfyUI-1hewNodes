# String Resolution Gemini 3.1 Flash Image - Gemini 3.1 Flash Image Resolution Selector

**Node Purpose:** `String Resolution Gemini 3.1 Flash Image` outputs the nearest Gemini 3.1 Flash Image resolution tier label (`0.5k` / `1k` / `2k` / `4k`) for each input image based on image area. If no image is connected, it passes through the manually selected label. It is designed to pair with Gemini 3.1 Flash Image.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `0.5k` / `1k` / `2k` / `4k` | Manual Gemini 3.1 Flash Image resolution label used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest resolution tier per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string; batched inputs are joined by newline in batch order. |

## Tiers and target areas

| Tier | 1:1 official size | Target area (pixels) |
| ---- | ----------------- | -------------------- |
| `0.5k` | `512x512` | 262,144 |
| `1k` | `1024x1024` | 1,048,576 |
| `2k` | `2048x2048` | 4,194,304 |
| `4k` | `4096x4096` | 16,777,216 |

## Features

- Gemini 3.1 Flash Image tiers: `0.5k` / `1k` / `2k` / `4k`, using each tier's official 1:1 size area as the target.
- Area-based matching: compares image area against the tier target areas and picks the nearest in log space.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.

## Typical Usage

- Convert image size into a resolution label before Gemini 3.1 Flash Image nodes that expect a resolution string, and connect it to the API node's `resolution` input.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- The target areas match the official 1:1 size of each Gemini 3.1 Flash Image tier (1k is 1024x1024, others scale by 0.5/1/2/4) item by item; that source belongs to 1hewOffice and its code is not in this repository.
- This node only outputs the string label and does not resize the image.
