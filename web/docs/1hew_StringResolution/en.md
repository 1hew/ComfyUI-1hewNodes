# String Resolution - Resolution Label Selector

**Node Purpose:** `String Resolution` outputs the nearest resolution tier label for each input image based on image area. If no image is connected, it passes through the manually selected label.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `0.5k` / `1k` / `2k` / `4k` | Manual resolution label used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest resolution tier per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string; batched inputs are joined by newline in batch order. |

## Features

- Area-based matching: compares image area against preset tiers `0.5k`, `1k`, `2k`, and `4k`.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.
- Lightweight helper: convenient for driving downstream model parameters with a plain string output.

## Typical Usage

- Convert image size into a resolution label before model/image nodes that expect `0.5k` / `1k` / `2k` / `4k`.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- This node only outputs the string label and does not resize the image.
