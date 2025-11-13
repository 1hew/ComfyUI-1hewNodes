# Image Edit Stitch

Stitches a reference image with an edit image (optionally with an edit mask),with configurable placement, spacing strip, and color parsing. Returns the composited image and two masks aligned to the output.

## Inputs

| Name | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| reference_image | IMAGE | Yes | - | The base image to stitch against. Supports batches. |
| edit_image | IMAGE | Yes | - | The image to attach to the reference. Supports batches. |
| edit_image_position | STRING (enum) | Yes | right | Where the edit image is attached: right, left, top, bottom. |
| match_edit_size | BOOLEAN | Yes | false | If true, reference is resized with padding to exactly match edit image size before stitching. If false, reference is resized keeping aspect ratio only along the relevant axis. |
| spacing | INT | Yes | 0 | Pixel width/height of the spacing strip inserted between images. 0 means no spacing strip. |
| spacing_color | STRING | Yes | "1.0" | Color of spacing strip. Supports advanced color strings (see below). |
| pad_color | STRING | Yes | "1.0" | Fill color used when resizing with padding (only applies when match_edit_size is true). Supports advanced color strings. |
| edit_mask | MASK | No | - | Optional edit mask aligned to edit_image. If omitted, a full-ones mask is assumed for the edit side. |

## Outputs

| Name | Type | Description |
| --- | --- | --- |
| image | IMAGE | The stitched image composed from reference and edit images, with optional spacing strip. Batch-safe. |
| mask | MASK | A mask aligned to the output image highlighting the edit_image area (1 over the edit region, 0 elsewhere including spacing). Batch-safe. |
| split_mask | MASK | A binary mask partitioning the output into reference (0) and edit (1) zones. Spacing strip area is 0. Batch-safe. |
