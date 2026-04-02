# Mask To Image - Convert Mask to RGB / RGBA Image

**Node Purpose:** `Mask To Image` converts an input `MASK` directly into an image output. You can choose the colors mapped from white and black mask regions, and optionally output RGBA with transparency.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask with grayscale falloff support |
| `fill_hole` | - | BOOLEAN | `False` | `True / False` | Fills enclosed black holes inside the mask before color mapping |
| `white_area_color` | - | STRING | `1.0` | same formats as `Image Solid` color: named / `#RRGGBB` / `r,g,b` / `0~1` grayscale / single-letter shorthand | Color mapped from white mask areas |
| `black_area_color` | - | STRING | `0.0` | same formats as `Image Solid` color: named / `#RRGGBB` / `r,g,b` / `0~1` grayscale / single-letter shorthand | Color mapped from black mask areas |
| `output_alpha` | - | BOOLEAN | `False` | `True / False` | When enabled, outputs RGBA and uses the original mask as the alpha channel while keeping the same RGB mapping |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Mapped RGB or RGBA image |

## Features

- Preserves grayscale transitions instead of forcing binary masks.
- RGB mode maps the mask as a gradient from `black_area_color` to `white_area_color`.
- RGBA mode keeps the same RGB mapping and writes the original grayscale mask into the alpha channel.
- `fill_hole` can be enabled to fill enclosed holes before mapping.
- Color parsing follows the same rules as the `Image Solid` node.

## Typical Usage

- Turn a grayscale mask into a visible color image.
- Create RGBA overlays with transparent background from a mask.
- Prepare brush / matte style inputs for downstream API or image-edit nodes.

## Notes & Tips

- When `output_alpha=true`, transparency comes directly from the original mask values.
- Disable `output_alpha` if you want a solid black-background color image, e.g. with `black_area_color=0.0`.
- Enable `output_alpha` if you want the mapped image plus an alpha channel driven by the original mask.
