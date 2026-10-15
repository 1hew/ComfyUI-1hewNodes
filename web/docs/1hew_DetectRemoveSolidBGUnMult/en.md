# Detect Remove SolidBG UnMult - Solid-background UnMult Keying

**Node Purpose:** `Detect Remove SolidBG UnMult` targets footage shot on a flat background: it estimates the background color from the image border, unmultiplies channel-wise to derive alpha, then optionally suppresses low alpha with `threshold` and composites the result onto `background_color`. It outputs both the foreground image and an alpha mask. Auto background estimation and dithering are fixed (auto background on, dither off), so no extra switches are exposed.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | required | IMAGE | - | - | Input image batch; 1 / 2 / 3 / 4 channels are accepted. |
| `threshold` | - | INT | `0` | `0-255` | 8-bit alpha black point. `0` disables cleanup; higher values suppress low alpha and remap the remaining alpha into `[0,1]`; `255` makes the output fully transparent. |
| `background_color` | - | STRING | empty | color name / HEX / RGB / `sm` / empty | Post-processing background. Empty keeps transparent RGBA; `sm` smart-selects black / white / `#808080` per image; any other value uses the `Image Alpha Split` color syntax. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The unmatted foreground: RGBA when `background_color` is empty, RGB composite otherwise. |
| `mask` | MASK | Float alpha mask in the 0~1 range (input alpha applied, `threshold` honored). |

## Features

- Background estimation: samples the border strips, votes on quantized joint-RGB clusters, and picks the most spatially supported color per image.
- UnMult: chooses one-sided or two-sided channel unmultiply based on whether the estimated background leans light or dark, then applies a gamut-safe unpremultiply so values stay in range.
- White-background gain: when the estimated background is nearly pure white, alpha receives an adaptive gain (median-driven, capped at 2.5) that improves translucent edges.
- `threshold` black point: remaps matte alpha with `(a - t) / (1 - t)` and zeroes pixels below the threshold while preserving any input alpha.
- Post compositing: `background_color` only affects the final image, never the background estimate; `sm` reuses `Image Alpha Split`'s smart background choice.

## Typical Usage

- Studio / flat-background keying: connect `image` and raise `threshold` from `0` to remove edge haze.
- Keep transparency for downstream compositing: leave `background_color` empty and use the RGBA `image` plus `mask`.
- Get a ready-to-use background immediately: set `background_color` to `sm`, `white`, and so on.

## Notes & Tips

- No model inference and no weight downloads; the node is pure tensor math.
- The background model assumes a single flat color, so gradients or multi-color backdrops may leave edge and translucency errors; `threshold` or `background_color` can mitigate them.
- Inputs that already carry alpha are supported as well: the input alpha takes part in the final mask.
- The `background_color` / `sm` syntax matches `Image Alpha Split` exactly, so the two nodes are interchangeable.
