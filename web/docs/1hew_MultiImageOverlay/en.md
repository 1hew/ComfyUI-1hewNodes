# Multi Image Overlay - Sequential Layer Compositing

**Node Purpose:** `Multi Image Overlay` overlays multiple image layers sequentially. RGBA inputs use the alpha channel for normal layer compositing, while RGB inputs are treated as opaque layers. If any input contains alpha, the node processes everything in RGBA and automatically outputs RGBA. Supports dynamic input ports `image_1..image_N`, multiple fit modes, and two batch modes (`parallel` / `stack`).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `fit_mode` | - | COMBO | `center` | `top_left` / `center` / `stretch` | Size fitting mode. |
| `color` | - | STRING | `1.0` | gray/HEX/RGB | Background color used when the composited result has no alpha (all-RGB inputs). |
| `batch_mode` | - | COMBO | `parallel` | `parallel` / `stack` | Batch handling. `parallel` composites corresponding batch indices frame-wise; `stack` flattens every batch item into a single layered preview. |
| `image_1..image_N` | - | IMAGE | - | - | Dynamic image layers where `image_1` is the topmost layer and larger indices go further down. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Composited image(s); a single preview in `stack` mode, or a batch in `parallel` mode. |

## Features

- Dynamic ports: inputs starting with `image_` automatically expand; `image_1` is the topmost layer and larger suffixes are placed further below.
- Size alignment: uses the bottom-most image dimensions as the base canvas size.
- Channel rule: if any input has alpha, all layers are composited in RGBA and the output automatically keeps alpha.
- Fit modes:
  - `center`: centers the overlay, cropping excess and leaving gaps transparent.
  - `top_left`: aligns the overlay to the top-left corner.
  - `stretch`: stretches the overlay to match the canvas size.
- Batch modes:
  - `parallel` (default): broadcasts smaller batches and composites corresponding indices in parallel, with `image_1` on top and `image_N` at the bottom.
  - `stack`: treats every item in every input batch as one layer and produces a single preview image; within each batch the first item is the top item.
- Alpha compositing: correctly handles premultiplied alpha blending for RGBA images. RGB images are treated as fully opaque.

## Typical Usage

- Image compositing: overlaying multiple elements with transparent backgrounds (e.g., stickers, watermarks, foreground subjects) onto a background image.
- Batch overlay: use `batch_mode=parallel` to composite corresponding frames across batches (broadcasting smaller batches to the maximum size).
- Single preview: use `batch_mode=stack` to flatten all frames of every input into one layered preview image.

## Notes & Tips

- Higher-numbered inputs are better suited for background/base layers, while `image_1` is better used for the top layer.
- To keep transparency for downstream nodes, make sure at least one input already carries alpha.
