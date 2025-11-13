# Image Three Stitch

Stitches three images by first combining image_2 and image_3, then attaching the result to image_1 according to the specified direction. Supports spacing strips and color parsing, with options for size matching or padding.

## Inputs

| Name | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| image_1 | IMAGE | Yes | - | The primary image to which the combined pair is attached. |
| image_2 | IMAGE | Yes | - | First member of the pair to be merged with image_3. |
| image_3 | IMAGE | Yes | - | Second member of the pair; combined with image_2 first. |
| direction | STRING (enum) | Yes | left | Where the combined pair is attached to image_1: top, bottom, left, right. |
| match_image_size | BOOLEAN | Yes | true | If true, sizes are matched along the stitching axis. If false, images are padded (no cropping) to align using pad_color. |
| spacing_width | INT | Yes | 10 | Spacing strip width inserted between the pair and between the pair and image_1. |
| spacing_color | STRING | Yes | "1.0" | Color of spacing strips. Supports color strings (see below). |
| pad_color | STRING | Yes | "1.0" | Padding color used when match_image_size is false. |

## Output

| Name | Type | Description |
| --- | --- | --- |
| image | IMAGE | The stitched result of three images, with optional spacing strips. |
