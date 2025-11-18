import torch
from comfy_api.latest import io


class StringCoordinateToBBoxMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringCoordinateToBBoxMask",
            display_name="String Coordinate to BBox Mask",
            category="1hewNodes/conversion",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("coordinates_string", default="", multiline=True),
                io.Combo.Input("output_mode", options=["separate", "merge"], default="merge"),
            ],
            outputs=[
                io.Mask.Output(display_name="bbox_mask"),
            ],
        )

    @classmethod
    def execute(cls, coordinates_string: str, image, output_mode: str) -> io.NodeOutput:
        batch_size, height, width, channels = image.shape
        if not coordinates_string.strip():
            return io.NodeOutput(torch.zeros((batch_size, height, width), dtype=torch.float32))
        lines = coordinates_string.strip().split("\n")
        bbox_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = line.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
            coords = []
            for part in line.replace(",", " ").split():
                try:
                    coords.append(int(float(part)))
                except ValueError:
                    continue
            if len(coords) >= 4:
                bbox_lines.append(coords[:4])
        if not bbox_lines:
            return io.NodeOutput(torch.zeros((batch_size, height, width), dtype=torch.float32))
        if output_mode == "separate":
            bbox_masks = []
            for bbox in bbox_lines:
                for b in range(batch_size):
                    bbox_mask = torch.zeros((height, width), dtype=torch.float32)
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    if x2 > x1 and y2 > y1:
                        bbox_mask[y1:y2, x1:x2] = 1.0
                    bbox_masks.append(bbox_mask)
            bbox_mask_tensor = torch.stack(bbox_masks)
        else:
            bbox_masks = []
            for b in range(batch_size):
                bbox_mask = torch.zeros((height, width), dtype=torch.float32)
                for bbox in bbox_lines:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    if x2 > x1 and y2 > y1:
                        bbox_mask[y1:y2, x1:x2] = 1.0
                bbox_masks.append(bbox_mask)
            bbox_mask_tensor = torch.stack(bbox_masks)
        return io.NodeOutput(bbox_mask_tensor)

