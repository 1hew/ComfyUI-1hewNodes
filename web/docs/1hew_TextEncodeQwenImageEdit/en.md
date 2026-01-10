# Text Encode QwenImageEdit - Qwen image-edit prompt encoder

**Node Purpose:** `Text Encode QwenImageEdit` builds a QwenImageEdit-compatible prompt with one or more image placeholders, resizes images for the vision encoder, and outputs a `CONDITIONING`. With an optional `VAE`, it also attaches `reference_latents` for image-reference guidance.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `clip` | - | CLIP | - | - | QwenImageEdit-compatible CLIP model used for tokenize and encode. |
| `vae` | optional | VAE | - | - | When connected, the node generates `reference_latents` for each provided image. |
| `image_1` | optional | IMAGE | - | - | First input image. The UI supports dynamic `image_2..image_10` ports. |
| `prompt` | - | STRING(multiline) | `""` | - | User instruction appended after the image placeholders. |
| `reference_skip_prep` | - | COMBO | `first` | `none` / `first` / `all` | Reference-latents strategy for preserving original size on 8-aligned inputs. |
| `reference_sq_area` | - | INT | 1024 | 64-8192 | Side length used to derive the target area (`reference_sq_area²`) for reference-latents normalization. |
| `vision_embed` | - | COMBO | `stretch` | `crop` / `pad` / `stretch` / `area` | Vision-image resize strategy used by `clip.tokenize(images=...)`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `conditioning` | CONDITIONING | Text/image conditioning. When `vae` is connected, `reference_latents` are appended into conditioning values. |

## Features

- Multi-image prompt: assembles `Picture 1..N` with Qwen vision placeholders and feeds them into `clip.tokenize(images=...)`.
- Vision resize modes: provides `crop`, `pad`, `stretch`, and `area` for stable vision-encoder input preparation.
- Reference latents: generates per-image VAE latents and attaches them to the conditioning for reference-based image editing.
- 8-alignment aware: preserves original VAE encode size on 8-aligned images according to `reference_skip_prep`.

## `vision_embed` Modes

- `area`: aspect-preserving resize to a target area of `384×384` pixels; output width/height follow `sqrt((384²)/(W×H))`.
- `crop`: aspect-preserving resize that covers `384×384`, then center-crop to `384×384`.
- `pad`: aspect-preserving resize that fits inside `384×384`, then center-pad to `384×384` with zeros.
- `stretch`: direct resize to `384×384`, allowing aspect change.

## `reference_latents` Rules

- Generation scope:
  - `none`: all images use area-normalized resize to `reference_sq_area²` and 8-alignment, then VAE encode.
  - `first`: the first image uses original size VAE encode when width/height align to 8; remaining images follow `none` behavior.
  - `all`: each image uses original size VAE encode when width/height align to 8; each remaining image follows `none` behavior.
- Normalization behavior (area + 8-alignment):
  - `scale = sqrt((reference_sq_area²) / (W×H))`
  - `W' = round((W×scale)/8)×8`, `H' = round((H×scale)/8)×8`
  - VAE encodes the resized RGB image (`:3` channels).

## Typical Usage

- Text-only conditioning: connect `clip`, set `prompt`, and keep `vae` empty.
- Image-edit conditioning: connect `clip`, connect one or more `image_*`, set `vision_embed`, and provide an instruction in `prompt`.
- Reference-guided editing: connect `vae` and tune `reference_skip_prep` plus `reference_sq_area` for reference-latents behavior.

## Notes & Tips

- Image ports are consumed in numeric order: `image_1`, `image_2`, `image_3`, ...
- For consistent vision behavior across images, use the same `vision_embed` mode for a workflow.
