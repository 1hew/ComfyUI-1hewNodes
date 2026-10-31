# Image Shift - Translation and Seam Mask

**Node Purpose:** `Image Shift` translates a whole image batch by `shift_x` / `shift_y` pixels. With `cyclic` enabled the pixels pushed out of the canvas wrap around to the opposite side, so the result tiles seamlessly; with `cyclic` disabled the uncovered area is filled with `background_color`. Alongside the shifted image it returns the shifted `mask` and a blurred seam `border_mask` (the original image border, i.e. the wrap-around seam), ready to feed an inpaint node.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Image batch to shift; 1, 3 and 4 (RGBA) channels are supported and preserved. |
| `shift_x` | - | INT | 256 | -9999–9999 | Horizontal translation; positive moves the content right. |
| `shift_y` | - | INT | 256 | -9999–9999 | Vertical translation; positive moves the content down. |
| `border_mask_width` | - | INT | 20 | 0–999 | Seam band width in pixels, clamped to half of the shorter side. |
| `border_mask_blur` | - | INT | 12 | 0–999 | Gaussian blur applied to the seam band after the shift. |
| `pad` | - | INT | 0 | 0–8192 | Wrap-pad the shifted canvas by this many pixels per side (`0` = off), so a seam that touches the border gains outside context before inpainting. |
| `cyclic` | - | BOOLEAN | true | - | Wrap the overflow around instead of filling it. |
| `background_color` | - | STRING | `1.0` | - | Fill for the uncovered area when `cyclic` is off. Same syntax as `Image Alpha Split`: 0–1 grey, `#RRGGBB`, colour names, single letters, `r,g,b`, or `sm` for the smart background. |
| `mask` | - | MASK | - | - | Companion mask shifted together with the image. It never changes the image, only the `mask` output; when unconnected that output is the alpha channel (RGBA) or a full-white mask. A 1-frame mask is broadcast and a differently sized mask is resized. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Shifted image batch, same size and channel count as the input. |
| `mask` | MASK | Shifted mask; it is the input `mask`, the alpha channel, or a full-white mask. With `cyclic` off the uncovered area is 0, so it doubles as a validity mask. |
| `border_mask` | MASK | Blurred seam band marking the original image border after the shift. |
| `pad_mask` | MASK | White frame (`1`) marking the pixels added by `pad`; all zeros when `pad = 0`. |

## Features

- Cyclic wrapping is implemented with `torch.roll` over the whole batch, so a 1024² image costs about a millisecond and stays on the GPU when the input does.
- Same colour input as `Image Alpha Split`: both nodes call the identical helper in `utils/_alpha_ops.py`, so `background_color` accepts the same strings, `sm` included.
- Pixel-exact non-cyclic bounds: a shift of `N` leaves exactly `N` fill pixels at the leading edge, and a zero shift is a true identity transform. (The LayerStyle reference leaves one extra pixel because of a strict `>` comparison.)
- Channel-faithful: RGBA input keeps its alpha, grayscale stays single-channel, and the fill colour is adapted to the channel count.
- Mask matching follows the batch: a single mask is broadcast, a full-batch mask is used per frame, a shorter batch reuses its last frame, and a differently sized mask is resized to the image.
- The seam band is built by direct slicing, so `border_mask_width` is the real pixel width (the reference builds it with `expand` + `resize`, which shrinks it by `W / (W + 2·width)` and makes it anisotropic on non-square canvases).
- The seam band depends only on canvas size and shift, so it is computed once and shared by the whole batch.
- Optional wrap padding: `pad > 0` extends the shifted `image`, `mask` and `border_mask` by `pad` pixels per side with an exact toroidal wrap, so the seam tips that sit on the canvas border become interior before inpainting.

## Background Colour (same as Image Alpha Split)

`background_color` is read by the same helper as `Image Alpha Split` (`utils/_alpha_ops.py`), so both nodes accept exactly the same strings:

| Form | Example | Notes |
| ---- | ------- | ----- |
| Grey (0–1) | `1.0`, `0.5`, `0.0` | `0` black … `1` white; the default is `1.0` |
| Hex | `#FF00FF`, `#0f0` | 3 or 6 digits |
| Name / single letter | `red`, `r`, `w` | PIL colour names; `r` red, `k` black |
| RGB | `128,192,255` or `0.5,0.7,0.9` | 0–255 and 0–1 forms are auto-detected |
| `sm` | - | Smart background: picks black, white or `#808080` per frame to keep the alpha subject readable, exactly like `Image Alpha Split`; an RGB input without alpha yields white |

Unknown values fall back to white, matching `Image Alpha Split`.

## Typical Usage

- **Seamless tiling:** shift by about half the width and height with `cyclic` on, set `pad` to about the seam band plus some margin (e.g. `320` for `border_mask_width = 100`) so the four arm tips that touch the canvas border get context from outside, inpaint with `border_mask`, crop back with `Image Edge Crop/Pad` (`uniform_amount = -pad`), then shift back or tile the result.
- **Layer offset:** shift a composited layer and use the `mask` output as its validity mask (non-cyclic mode fills the gap with `background_color`).
- **Texture scroll:** animate `shift_x` while `cyclic` is on to scroll a tileable texture without a visible border.

## Notes & Tips

- Keep `border_mask_blur` at or below half of `border_mask_width`; a much larger blur flattens the band below 50% and the inpaint mask stops being useful.
- In cyclic mode the seam is where the original left/right (or top/bottom) edges meet, which is exactly where `border_mask` sits; in non-cyclic mode the band still marks the original border, not the new edge of the covered content.
- Shifts larger than the canvas are allowed: cyclic wraps modulo the size, non-cyclic returns only `background_color`.
- The seam cross necessarily touches all four canvas edges: those four tips are where the seam continues into the neighbouring tile, so inpaint them with wrap padding (`pad > 0`) instead of on the bare canvas, otherwise the tips are guessed from one side and the tile shows a mismatch.
- `pad` should cover the seam band plus context: at a half offset the band is `2 × border_mask_width`, so `border_mask_width = 100` needs `pad ≥ 200` (use `320` for margin). Larger values only cost memory.
