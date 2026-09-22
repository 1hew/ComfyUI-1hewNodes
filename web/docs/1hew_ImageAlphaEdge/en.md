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
- Design-software feathering (Photoshop-style): `edge` and `feather` modify the alpha channel only. The subject's RGB is never blurred, dilated or recoloured, so fine multi-colour details (fireworks, hair, smoke) keep their exact colours instead of turning into Voronoi facets or opaque bands. Only pixels that were fully transparent and become newly visible get a smooth, alpha-aware colour extension so the feathered band stays clean when composited.
- Fixed pipeline order: shrink/expand → feather (alpha only) → colour extension into newly visible pixels.
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
- The colour extension is a fixed pipeline step with no toggle: it runs only when `feather > 0`. The post-edge alpha is the colour donor, so a fringe removed by a negative `edge` cannot be sampled again. Only pixels that were fully transparent in the input but become visible through edge adjustment or feathering receive colour; existing translucent detail keeps its RGB unchanged.
- The colour field uses the same Gaussian scale as `feather`, so a large feather does not expose old-background RGB stored in transparent input pixels. It rewrites RGB only in that new transition band; composite the original back afterwards if those original transparent-pixel RGB values must be retained.
- Hole filling is out of scope: this node only moves the alpha level sets, it never repairs interior holes. Run a mask node first if you need holes filled, then feed the result here.
- Large `feather` values cost more time; on a 2048×2048 image a radius up to 256 still completes in well under a second.
- Inputs without an alpha channel return a fully opaque `mask` and an RGBA image, so the node is safe to insert anywhere.
