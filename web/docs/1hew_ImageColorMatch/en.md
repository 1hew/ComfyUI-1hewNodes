# Image Color Match - Transfer Reference Colors

**Node Purpose:** `Image Color Match` transfers the color characteristics of `reference_image` onto `source_image` using one of several color-transfer algorithms. `wavelet` and `adain` are implemented natively; `mkl`, `hm`, `reinhard`, `mvgd` and the hybrids require the `color-matcher` package. Batches are supported, and reference frames are cycled (`i % ref_count`) when counts differ.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `source_image` | - | IMAGE | - | - | Image that will be recolored. |
| `reference_image` | - | IMAGE | - | - | Reference image whose colors are transferred. |
| `method` | - | COMBO | `mkl` | `wavelet` / `adain` / `mkl` / `hm` / `reinhard` / `mvgd` / `hm-mvgd-hm` / `hm-mkl-hm` | Color-transfer algorithm. `wavelet`/`adain` run natively; the others require `color-matcher`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Recolored image batch; alpha is preserved when the source has one. |

## Features

- Multiple algorithms: `wavelet` and `adain` are pure-PyTorch (ported from Easy-Use); `mkl`, `hm`, `reinhard`, `mvgd`, `hm-mvgd-hm`, `hm-mkl-hm` delegate to the `color-matcher` package.
- Batch cycling: a single reference applies to all frames; equal batches match frame-wise; smaller batches repeat; larger batches truncate.
- Alpha preservation: color transfer operates on RGB and keeps the source alpha channel untouched.

## Typical Usage

- Recolor a generated image to match a reference photo's palette.
- Harmonize a composite so the subject matches the background tone.

## Notes & Tips

- Methods other than `wavelet`/`adain` require installing `color-matcher` (`pip install color-matcher`); the node raises a clear error otherwise.
- Reference frames are cycled by index, matching the project's broadcast/cycling convention.
