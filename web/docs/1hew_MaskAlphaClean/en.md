# Mask Alpha Clean - Clean Noise in Mask Alpha

**Node Purpose:** `Mask Alpha Clean` removes weak alpha speckles and tiny disconnected regions from mask batches with a simple strength preset interface.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch; 2D masks are expanded to `[B,H,W]`. |
| `clean_strength` | - | COMBO | `balanced` | `soft / balanced / strong` | Preset controlling threshold and small-island filtering strength. |
| `detect_only` | - | BOOLEAN | False | - | When `True`, returns original mask and outputs only detection in `noise_mask`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Cleaned mask batch (or unchanged when `detect_only=True`). |
| `noise_mask` | MASK | Detected noisy regions mask (`1` = suspected noise). |

## Features

- Simple preset workflow: `soft`, `balanced`, `strong`.
- Removes weak alpha fragments and tiny isolated components.
- Includes detection-only mode for safe inspection.

## Typical Usage

- Clean segmentation masks before crop/paste/composite operations.
- Remove tiny floating mask dots caused by matting/refinement artifacts.
- Validate cleanup impact with `detect_only` before final output.

## Notes & Tips

- Use `soft` when masks contain delicate thin structures.
- Use `strong` for aggressively denoising noisy matte outputs.
