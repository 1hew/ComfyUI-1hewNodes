# Match Brightness Contrast - Match brightness and contrast

**Node Purpose:** `Match Brightness Contrast` adjusts the brightness and contrast of `source_image` to match `reference_image`. It supports histogram matching and standard mean/std matching, plus optional edge-only statistics and sequence consistency modes for batch workflows.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `source_image` | - | IMAGE | - | - | Source image batch to be adjusted. |
| `reference_image` | - | IMAGE | - | - | Reference image batch that provides target tone distribution. |
| `edge_amount` | - | FLOAT | 0.2 | 0.0-8192.0 | Statistics area control. `<=0`: full image; `<1.0`: edge ratio based on the shorter side; `>=1.0`: edge width in pixels. |
| `consistency` | - | COMBO | `lock_first` | `lock_first` / `lock_mid` / `lock_end` / `frame_match` | Sequence consistency. `lock_*` computes parameters once from a selected pair and applies to all source frames; `frame_match` computes per frame. |
| `method` | - | COMBO | `histogram` | `standard` / `histogram` | Matching method. `histogram` uses per-channel CDF mapping; `standard` uses mean/std linear remap. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Adjusted image batch |

## Features

- **Matching Algorithms**:
  - `histogram`: Mapping based on Cumulative Distribution Function (CDF) of channel histograms, reproducing reference tone distribution more precisely.
  - `standard`: Linear transformation based on mean and standard deviation, preserving more source texture features, yielding softer results.
- **Area Control (`edge_amount`)**:
  - 0: Statistics from the full image.
  - >0: Statistics only from the peripheral edge areas, ignoring the center. Useful for matching ambient atmosphere while ignoring subject differences.
- **Sequence Consistency (`consistency`)**:
  - `lock_first`: Computes parameters from `source_image[0]` and `reference_image[0]`.
  - `lock_mid`: Computes parameters from the middle frame of each input batch.
  - `lock_end`: Computes parameters from the last frame of each input batch.
  - `frame_match`: Computes parameters for each frame pair (`reference_image[i % ref_batch]`).

## Typical Usage

- **Video Tone Unification**: Connect video frames to `source_image` and a tone reference sequence to `reference_image`, set `consistency=lock_mid` or `lock_end` to keep batch tone stable when the early reference frames vary.
- **Compositing Blending**: For image compositing, use foreground as `source_image` and background as `reference_image`, set appropriate `edge_amount` (e.g., 0.2) to blend foreground edges with the background tone.

## Notes & Tips

- For sequence workflows, `lock_*` keeps a single mapping across frames, which helps keep tone consistent.
- `edge_amount` focuses statistics on the border region and supports stable background tone matching.
