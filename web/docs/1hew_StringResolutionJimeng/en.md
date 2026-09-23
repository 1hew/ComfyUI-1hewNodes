# String Resolution Jimeng - Jimeng Resolution Selector

**Node Purpose:** `String Resolution Jimeng` outputs the nearest Jimeng resolution tier label (`1k` / `2k` / `4k`) for each input image based on image area. If no image is connected, it passes through the manually selected label. It is designed to pair with the Jimeng node family (jimeng_image_30 etc. in ComfyUI-1hewJimeng) and `Image Resize Jimeng` in this repository.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1k` | `1k` / `2k` / `4k` | Manual Jimeng resolution label used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest resolution tier per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string; batched inputs are joined by newline in batch order. |

## Tiers and target areas

| Tier | 1:1 official size | Target area (pixels) |
| ---- | ----------------- | -------------------- |
| `1k` | `1328x1328` | 1,763,584 |
| `2k` | `2048x2048` | 4,194,304 |
| `4k` | `4096x4096` | 16,777,216 |

## Tiers available per node / model

The Jimeng `resolution` options are defined per node and are not uniform; this node outputs their union, so pick a label the paired node actually supports:

| Jimeng node | Available tiers |
| ----------- | --------------- |
| `Jimeng Image T2I` | `1k` / `2k` / `4k` |
| `Jimeng Image Multi Edit` | `2k` / `4k` |
| `Jimeng Image 4.0` / `4.1` / `4.5` / `4.6` / `4.7` / `5.0 Lite` | `2k` / `4k` |
| `Jimeng Image 3.0` / `ControlNet` / `IP` | `1k` / `2k` |
| `Jimeng Image Single Edit` / `Style` / `Subject` | `1k` / `2k` |

- Model `2.0_pro` is not tiered by `1k` / `2k` / `4k`; it uses a separate 1024-base size table (`2.0_pro_ratios` in `config.json`), so the `resolution` label does not change the size group there. That is why it is outside this node's scope.

## Features

- Jimeng tiers: `1k` / `2k` / `4k`, using each tier's official 1:1 size area as the target.
- Area-based matching: compares image area against the tier target areas and picks the nearest in log space.
- Manual fallback: returns the selected label directly when `image` is empty.
- Batch support: outputs one resolution label per image in the batch.

## Typical Usage

- Convert image size into a resolution label before Jimeng nodes that expect a resolution string, and connect it to the API node's `resolution` input.
- Keep manual control by leaving `image` unconnected and selecting a fixed label.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- The target areas match the official 1:1 size of each Jimeng tier (the `1:1` entries of `1k_ratios` / `2k_ratios` / `4k_ratios` in `ComfyUI-1hewJimeng`'s `config.json`, read by `get_ratio_dimensions()` in `core/utils_common.py`) item by item; that source belongs to the ComfyUI-1hewJimeng plugin and its code is not in this repository.
- This node only outputs the string label and does not resize the image.
