# Detect Yolo

**Node Function:** The `Detect Yolo` node performs object detection using YOLO models. Supports multiple versions including YOLOv5, YOLOv8, YOLOv9, YOLOv10, and YOLOv11. It can detect objects in images, generate masks for detected objects, and create visualization images with bounding boxes and labels. Features threshold control, mask selection by index, customizable label display, and intelligent label scaling functionality.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image tensor for object detection |
| `yolo_model` | Required | STRING | - | Available .pt files | YOLO model file (.pt format), supports multiple versions |
| `threshold` | Required | FLOAT | 0.3 | 0.0-1.0 | Detection confidence threshold |
| `mask_index` | Required | STRING | "-1" | - | Comma-separated indices for mask selection (-1 or empty = all) |
| `label` | Required | BOOLEAN | True | True/False | Whether to display labels on detection results |
| `label_size` | Required | FLOAT | 1.0 | 0.1-5.0 | Label size scaling factor, controls font size and border thickness |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `plot_image` | IMAGE | Visualization image with bounding boxes and optional labels |
| `mask` | MASK | Combined mask of selected detected objects |
