# Image Alpha Clean - Clean Alpha Edge Noise in RGBA

**Node Purpose:** `Image Alpha Clean` removes tiny alpha artifacts and weak semi-transparent edge noise from image alpha channels, helping avoid speckles when applying strokes/shadows in external editors.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch; RGBA is recommended. |
| `clean_strength` | - | COMBO | `balanced` | `soft / balanced / strong` | Preset controlling alpha threshold and tiny-island removal strength. |
| `detect_only` | - | BOOLEAN | False | - | When `True`, only detects noise regions and keeps input image unchanged. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Cleaned RGBA image batch (or unchanged when `detect_only=True`). |
| `noise_mask` | MASK | Detected noise regions mask (`1` = suspected noise). |

## Features

- Preset-driven cleanup with minimal parameters for daily workflows.
- Removes weak alpha fragments and tiny disconnected alpha islands.
- Optional edge color bleed to reduce fringe artifacts near transparency.
- Supports inspection mode via `detect_only` before actual cleanup.

## Typical Usage

- Pre-clean PNG/RGBA assets before Photoshop stroke or layer effects.
- Stabilize alpha boundaries before compositing and mask-based blending.
- Inspect problematic edge noise by reading `noise_mask`.

## Notes & Tips

- `soft` preserves more edge detail; `strong` removes more tiny alpha residues.
- For RGB inputs without alpha, this node treats alpha as fully opaque.
