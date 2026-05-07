# Image List Interleave - Segment-wise Interleaved Image List Reorder

**Node Purpose:** `Image List Interleave` splits the input image list into `segment_count` contiguous segments, then rebuilds the list in column-first order: first image of each segment, then second image of each segment, and so on. It only changes item order and does not normalize sizes, making it suitable for reordering and saving images with different dimensions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE / IMAGE_LIST | - | - | Input image data; accepts a batch or list. Batch input is split into single-image list items. |
| `segment_count` | - | INT | `2` | `1-100000` | Number of contiguous segments to split before interleaving. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_list` | IMAGE_LIST | Reordered image list. |

## Features

- Contiguous split: divides the list into `segment_count` consecutive segments in original order.
- Column-first interleave: outputs the 1st item of each segment first, then the 2nd item of each segment, and so on.
- Size preservation: no padding, cropping, or resizing; every image keeps its original dimensions.
- Input compatibility: batch input is split into single `[1,H,W,C]` items, while list input is recursively flattened.

## Typical Examples

- Input `1, 2, 3, 4, 5, 6`, `segment_count=3`
  - Segments: `(1, 2)`, `(3, 4)`, `(5, 6)`
  - Output: `1, 3, 5, 2, 4, 6`

- Input `1, 2, 3, 4, 5, 6, 7`, `segment_count=3`
  - Segments: `(1, 2, 3)`, `(4, 5)`, `(6, 7)`
  - Output: `1, 4, 6, 2, 5, 7, 3`

## Notes & Tips

- When `segment_count=1`, the output order stays unchanged.
- When `segment_count` is larger than the input count, it is effectively capped by the input count, so each segment has at most one image and the output remains in original order.
- Prefer this node over the batch version when images have different sizes and should keep their dimensions before `Save Image`.
