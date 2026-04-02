# Multi Image Overlay - Sequential Layer Compositing

**Node Purpose:** `Multi Image Overlay` overlays multiple image layers sequentially. RGBA inputs use the alpha channel for normal layer compositing, while RGB inputs are treated as opaque layers. If any input contains alpha, the node processes everything in RGBA and automatically outputs RGBA. Supports dynamic input ports `image_1..image_N` and multiple fit modes.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `fit_mode` | - | COMBO | `center` | `top_left` / `center` / `stretch` | Size fitting mode. |
| `color` | - | STRING | `1.0` | - | Legacy background color parameter kept for compatibility; when any input has alpha, the node outputs RGBA directly. |
| `image_1..image_N` | - | IMAGE | - | - | Dynamic image layers where `image_1` is the topmost layer and larger indices go further down. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch of composited images. |

## Features

- Dynamic ports: inputs starting with `image_` automatically expand; `image_1` is the topmost layer and larger suffixes are placed further below.
- Size alignment: uses the bottom-most image dimensions as the base canvas size.
- Channel rule: if any input has alpha, all layers are composited in RGBA and the output automatically keeps alpha.
- Fit modes:
  - `center`: centers the overlay, cropping excess and leaving gaps transparent.
  - `top_left`: aligns the overlay to the top-left corner.
  - `stretch`: stretches the overlay to match the canvas size.
- Alpha compositing: correctly handles premultiplied alpha blending for RGBA images. RGB images are treated as fully opaque.

## Typical Usage

- Image compositing: overlaying multiple elements with transparent backgrounds (e.g., stickers, watermarks, foreground subjects) onto a background image.
- Batch overlay: supports batch processing; broadcasts smaller batches to match the maximum batch size.

## Notes & Tips

- Higher-numbered inputs are better suited for background/base layers, while `image_1` is better used for the top layer.
- To keep transparency for downstream nodes, make sure at least one input already carries alpha.