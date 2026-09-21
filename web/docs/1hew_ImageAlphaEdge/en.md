# Image Alpha Edge - Shrink / expand and feather the alpha edge

**Node Purpose:** `Image Alpha Edge` adjusts the opaque region of an image that has an alpha channel: it shrinks that boundary inward or expands it outward, then feathers the result. The offset runs as grayscale morphology, so the matte's own anti-aliasing survives intact. It is built for cleaning up matte fringe after background removal — shrinking inward cuts away the semi-transparent boundary pixels that are contaminated by the old background, feathering rebuilds a smooth transition, and a built-in RGB bleed pushes the opaque color into that band so it does not show a dark or colored halo.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image, single or batch. The alpha channel is used when present (>= 4 channels); RGB/gray inputs are treated as fully opaque. |
| `edge` | - | INT | 0 | -256–256 (step 1) | Signed edge adjustment in pixels: negative shrinks inward, positive expands outward. 0 = no edge adjustment. Implemented as grayscale min/max filtering, which keeps the matte's sub-pixel transition. |
| `feather` | - | FLOAT | 0.0 | 0–256 (step 0.1) | Gaussian feather radius (sigma) applied to the alpha after the edge adjustment. |

## Why the offset has no jaggies

`edge` runs min/max filtering directly on the **grayscale alpha** instead of binarizing it first:

- for a locally linear soft edge (segmentation mattes are typically a 1–3 px ramp), a disk-shaped min filter of radius r is exactly a **shift of that ramp by r pixels**;
- so every alpha level set moves by r pixels — the edge is taken off while the sub-pixel information in the transition band is preserved;
- for a hard 0/1 matte the result is **pixel-for-pixel identical** to the old binary morphology; there is simply no anti-aliasing to preserve.

Measured on a circle of radius 40 with `edge = -5`, `feather = 0` (ideal radius 35):

| Approach | contour jitter | max deviation from the ideal circle | grey levels |
| -------- | -------------- | ----------------------------------- | ----------- |
| binarize + erode (old) | 0.0430 | 0.77 px | 2 |
| grayscale min filter (now) | **0.0034** | **0.39 px** | 69 |

Contour jitter is the mean change of the contour radius between adjacent angles — lower is smoother.

If the input really is a hard-edged matte the result stays hard-edged; use `feather` to rebuild a transition band.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGBA image whose alpha has been shrunk/expanded and feathered. |
| `mask` | MASK | The processed alpha, for reuse by mask nodes. |

## Features

- Anti-aliased: `edge` is grayscale morphology, so the matte's sub-pixel transition survives (the old binarization turned soft edges into a 1 px staircase).
- Signed edge control: one parameter covers both inward shrink (negative) and outward expand (positive).
- Fringe removal: with `edge < 0` the contaminated boundary band is discarded rather than blended, which is the reliable way to kill matte fringe.
- Built-in decontamination: before feathering, the colour of the nearest **solidly opaque** pixel is pushed into the transition band (fixed behaviour, no toggle). The source requires `alpha >= 0.9`, so a soft matte's own contaminated ramp does not keep spreading the old background colour outward.
- Fixed pipeline order: shrink/expand → RGB bleed → feather.
- Soft matte friendly: with `edge = 0` the original soft alpha is kept (no binarization at all).
- Single responsibility: edges only — hole filling is matte repair and belongs to a dedicated mask node.
- Batch support: image batches are processed frame by frame with synchronized `image` and `mask` outputs.
- Pass-through: with `edge = 0` and `feather = 0` the input is returned unchanged (no colour extension happens either).

## Typical Usage

- Defringe after background removal: set `edge` to about `-2`–`-5` and `feather` to `1`–`3` to trim the contaminated edge while keeping a soft transition.
- Weak outer halo on a soft edge: just shrink with `edge = -1`–`-2`.
- Keep the silhouette from growing: make the shrink at least about twice the feather so the feathered band stays inside the original outline.
- Grow a selection: use a positive `edge` to expand the opaque region outward before feathering.
- Keep the original soft edge: `edge = 0`, tuning `feather` only.

## Notes & Tips

- `edge` does not harden a soft matte: grayscale morphology keeps the transition band, so hair, smoke and motion blur can be shrunk directly. A hard matte stays hard after the offset — use `feather` to build a transition band.
- The colour extension is a fixed pipeline step with no toggle: it runs only when `feather > 0`, takes its colours from the solid region (`alpha >= 0.9`) after the edge adjustment, and fills only the band the feather blur can reach (about `4 x feather` pixels) - transparent pixels further away keep their original RGB. If the matte never reaches 0.9 the extension is skipped rather than spreading contaminated colour.
- It rewrites RGB data: inside the transition band, transparent pixels that held the old background colour are replaced by the nearest subject colour. If you must keep the original RGB there, composite the original image back afterwards.
- Turning it off would not be "cleaner": the feathered band would simply show the old background colour as a halo. Measured on a red subject over a blue background, the blue leak with no extension peaked at +0.23. With the corrected colour source the leak measures 0.0000 for soft edges from 1 px to 10 px wide.
- Hole filling is out of scope: this node only moves the alpha level sets, it never repairs interior holes. Run a mask node first if you need holes filled, then feed the result here.
- Large `feather` values cost more time; on a 2048×2048 image a radius up to 256 still completes in well under a second.
- Inputs without an alpha channel return a fully opaque `mask` and an RGBA image, so the node is safe to insert anywhere.
