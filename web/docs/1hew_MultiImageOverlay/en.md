# Multi Image Overlay - Sequential Layer Compositing

**Node Purpose:** `Multi Image Overlay` overlays multiple image layers sequentially. RGBA inputs use the alpha channel for normal layer compositing, while RGB inputs are treated as opaque layers. Supports dynamic input ports `image_1..image_N`, multiple fit modes, and background color settings.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `fit_mode` | - | COMBO | `center` | `top_left` / `center` / `stretch` | Size fitting mode. |
| `color` | - | STRING | `1.0` | - | Background color, supports various formats (e.g., hex, RGB, color names). |
| `output_alpha` | - | BOOLEAN | `false` | - | Whether to output an image with an alpha channel. |
| `image_1..image_N` | - | IMAGE | - | - | Dynamically input image layers, overlaid in numerical order. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch of composited images. |

## Features

- Dynamic ports: inputs starting with `image_` automatically expand and are overlaid from bottom to top based on their numerical suffix (`image_1` -> `image_2` -> ...).
- Size alignment: uses the dimensions of the first image (`image_1`) as the base canvas size.
- Fit modes:
  - `center`: centers the overlay, cropping excess and leaving gaps transparent.
  - `top_left`: aligns the overlay to the top-left corner.
  - `stretch`: stretches the overlay to match the canvas size.
- Alpha compositing: correctly handles premultiplied alpha blending for RGBA images. RGB images are treated as fully opaque.
- Background fill: when `output_alpha=false`, fills the transparent background of the final image with the specified `color`.

## Typical Usage

- Image compositing: overlaying multiple elements with transparent backgrounds (e.g., stickers, watermarks, foreground subjects) onto a background image.
- Batch overlay: supports batch processing; broadcasts smaller batches to match the maximum batch size.

## Notes & Tips

- `image_1` determines the final output dimensions; it is usually recommended to connect the background or main image to `image_1`.
- Enable `output_alpha` if you need to preserve transparency for downstream nodes.