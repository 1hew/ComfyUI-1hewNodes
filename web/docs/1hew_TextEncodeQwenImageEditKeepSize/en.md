# Text Encode QwenImageEdit Keep Size - Vision-language conditioning with size policies

**Node Purpose:** `Text Encode QwenImageEdit Keep Size` encodes a text prompt together with one or more input images into QwenImageEdit-compatible conditioning. Each image contributes a vision token stream (fixed at 384×384), and optional VAE reference latents are attached according to a keep-size policy. This enables image-grounded editing while preserving composition.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `clip` | - | CLIP | - | - | CLIP model that supports vision tokens and scheduled encoding. |
| `prompt` | - | STRING | `` | multiline | Text instruction appended after image vision tokens. |
| `keep_size` | - | COMBO | `first` | `none` / `first` / `all` | Policy for VAE reference latents: `none` scales all to target area; `first` preserves the first image if already multiple-of-8; `all` preserves any image that already satisfies multiples-of-8. |
| `base_size` | - | COMBO | `1024` | `1024` / `1536` / `2048` | Target area side; when not preserving size or when the input size is non-multiple-of-8, latents are scaled so that width×height ≈ `base_size²` and rounded to multiples of 8. |
| `vae` | optional | VAE | - | - | VAE used to encode RGB reference latents attached to conditioning. |
| `image_1` | optional | IMAGE | - | - | First image; contributes vision tokens and may produce reference latents. |
| `image_2…image_N` | optional | IMAGE | - | - | Additional images recognized by numeric suffix; ordered by suffix. Up to 10 images recommended. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `conditioning` | CONDITIONING | QwenImageEdit-compatible conditioning with optional `reference_latents` attached. |

## Features

- Vision tokens at 384×384: each input image is resized to `384×384` (area resample) for the CLIP tokenize step; the prompt is prefixed by image token placeholders and wrapped in a Qwen-compatible llama template.
- Size-preserving latents: `keep_size` controls whether to retain original spatial size when already multiples-of-8, or to scale to a target area derived from `base_size` and round width/height to multiples of 8.
- RGB-only latents: reference latents are encoded from RGB channels; alpha is omitted to keep semantics consistent.
- Dynamic ordering: numeric-suffix inputs (`image_1`, `image_2`, …) are collected and ordered by suffix, ensuring stable multi-image guidance.
- Scheduled encoding: tokens are fed to the CLIP for scheduled encoding; VAE latents are attached via `reference_latents` for downstream use.

## Typical Usage

- Anchor editing: provide one or more images to ground the edit and set `keep_size=first` to preserve the first anchor’s composition when it already meets multiples-of-8.
- Preserve all anchors: set `keep_size=all` to retain original sizes for all anchors that meet multiples-of-8; others are scaled to the selected `base_size` area.
- Unify latent area: set `keep_size=none` and choose `base_size=1536` or `2048` to standardize reference latents for consistent model behavior.
- Text-only conditioning: omit images to encode pure text; add images later to enrich guidance.

## Notes & Tips

- Multiples-of-8 constraint: when preserving size, width and height must be divisible by 8; otherwise the node scales to the chosen `base_size²` area and rounds to multiples of 8.
- Channel handling: VAE reference latents use RGB channels; vision tokens include any channels but are internally processed from the resized image tensor.
- Ordering: images are collected by `image_*` numeric suffix; gaps are allowed and order follows ascending suffix numbers.
- Device safety: all tensors preserve dtype/device; batched inputs are supported in channels-last layout (`B×H×W×C`).