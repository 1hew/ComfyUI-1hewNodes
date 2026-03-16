# Detect Remove BG Refine - RMBG Mask Post-Refinement

**Node Purpose:** `Detect Remove BG Refine` is designed as a post-processing node for `RMBG-1.4`-style outputs. It takes the **original image** plus model `mask`, performs alpha refinement, anti-aliasing, optional vector hard-edge shaping, and decontamination, then outputs optimized RGBA and refined mask.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | required | IMAGE | - | - | Original source image (must be the original, not model-processed image). |
| `mask` | required | MASK | - | - | Alpha mask produced by RMBG model output. |
| `type` | - | COMBO | `bitmap` | `bitmap` / `vector` | Edge mode: soft bitmap edges or harder vector-like contour edges. |
| `subject_protect` | - | FLOAT | 0.85 | 0.0~1.0 | Foreground core protection strength. |
| `feather` | - | FLOAT | 1.0 | 0.0~64.0 | Feather control (mapped internally to smoothing intensity). |
| `decolor_edge` | - | FLOAT | 1.0 | 0.0~1.0 | Edge decontamination strength. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Standard RGBA output (alpha as the 4th channel, no premultiply). |
| `mask` | MASK | Refined alpha mask. |

## Features

- Alpha refinement first: bilateral/Gaussian smoothing, foreground-core protection, and edge anti-aliasing.
- Optional vector hardening: `vector` mode reconstructs harder contour shapes after refinement.
- Fixed background estimation policy: white/black priority with auto fallback.
- Decontamination always uses original RGB with refined alpha.

## Typical Usage

- RMBG polishing: feed `mask` from `Detect Remove BG`, and pass the original image to this node.
- Hard-edge assets (logos/graphics): use `type=vector` and increase `subject_protect` if needed.
- Natural portrait edges: use `type=bitmap` with moderate `feather` and `decolor_edge`.

## Notes & Tips

- Do not feed model-processed images back into this node; it can amplify edge pollution and color shifts.
- Internal policy is fixed by design; there are no extra `bg_mode`, `min_area_ratio`, or `premultiply` parameters.
