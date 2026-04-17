# Image Batch Interleave - segment-wise interleaved batch reorder

**Node Purpose:** `Image Batch Interleave` splits the input image batch into `segment_count` contiguous segments, then rebuilds the batch in column-first order: first image of each segment, then second image of each segment, and so on.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch `(B, H, W, C)` |
| `segment_count` | - | INT | `2` | `1-100000` | Number of contiguous segments to split before interleaving |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Reordered image batch |

## Features

- Contiguous split: divides the batch into `segment_count` consecutive segments in original order.
- Column-first interleave: outputs the 1st item of each segment first, then the 2nd item of each segment, and so on.
- Balanced segmentation: when the batch size is not divisible by `segment_count`, earlier segments receive one extra item so no image is dropped.
- Pure reorder: only changes batch index order; pixel values are untouched.

## Typical Examples

- Input `1, 2, 3, 4, 5, 6`, `segment_count=3`
  - Segments: `(1, 2)`, `(3, 4)`, `(5, 6)`
  - Output: `1, 3, 5, 2, 4, 6`

- Input `1, 2, 3, 4, 5, 6, 7`, `segment_count=3`
  - Segments: `(1, 2, 3)`, `(4, 5)`, `(6, 7)`
  - Output: `1, 4, 6, 2, 5, 7, 3`

## Notes & Tips

- When `segment_count=1`, the output order stays unchanged.
- When `segment_count` is larger than the batch size, it is effectively capped by the batch size, so each segment has at most one image and the output remains in original order.
- This node is useful when you want to redistribute a sequential batch into a cross-segment round-robin order for preview, stitching, or timing adjustments.
