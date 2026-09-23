# String Resolution GPT Image 2.0 - GPT Image 2.0 Resolution Selector

**Node Purpose:** `String Resolution GPT Image 2.0` outputs the nearest GPT Image 2.0 resolution tier label (`1k` / `2k` / `4k`) for each input image based on image area. If no image is connected, it passes through the manually selected label. It is designed to pair with GPT Image 2.0.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `1k` / `2k` / `4k` | Manual GPT Image 2.0 resolution label used when `image` is not connected. |
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
| `4k` | `3840x2160` (1:1 is `2880x2880`) | 8,294,400 |

## Features

- GPT Image 2.0 tiers: `1k` / `2k` / `4k`, using each tier's official 1:1 size area as the target.
- Area-based matching: compares image area against the tier target areas and picks the nearest in log space.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.

## Typical Usage

- Convert image size into a resolution label before GPT Image 2.0 nodes that expect a resolution string, and connect it to the API node's `resolution` input.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- The target areas match `GPTImage20.MAX_PIXELS_BY_RESOLUTION` in 1hewOffice (1k=1024², 2k=2048², 4k=3840x2160, the same area as the 4k 1:1 size 2880x2880) item by item; that source belongs to 1hewOffice and its code is not in this repository.
- This node only outputs the string label and does not resize the image.
