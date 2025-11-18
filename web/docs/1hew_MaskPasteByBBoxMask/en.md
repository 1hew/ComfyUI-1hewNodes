# Mask Paste by BBox Mask - Paste Mask into Bounding Box

**Node Purpose:** `Mask Paste by BBox Mask` pastes a `paste_mask` into the bounding box region defined by `bbox_mask` on top of an optional `base_mask` (defaults to zeros). Handles batch cycling and resizes the paste to fit the bbox.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `paste_mask` | - | MASK | - | - | Mask to paste into the bbox.
| `bbox_mask` | - | MASK | - | - | Mask whose non-zero region defines the bbox.
| `base_mask` | optional | MASK | - | - | Destination mask; defaults to zeros matching `bbox_mask` when absent.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Mask after pasting into the bbox region.

## Features

- Batch cycling: aligns `base_mask`, `paste_mask`, and `bbox_mask` across differing batch sizes via modulo indexing.
- BBox detection: threshold-based bbox from `bbox_mask` (`>10` on `[0..255]`).
- Size fit: resizes `paste_mask` to bbox dimensions with Lanczos and pastes at bbox location.
- Robust fallback: when bbox is absent, returns the `base_mask` item as-is.

## Typical Usage

- Region replacement: replace or insert a refined mask region within a detected bbox.
- Crop-paste workflows: pair with mask cropping nodes to transform and relocate mask content.

## Notes & Tips

- Ensure `paste_mask` semantics match `base_mask` (white=selected) to avoid unintended inversions.
- For precise alignment, `paste_mask` should be prepared to match bbox content (e.g., from a cropped mask).