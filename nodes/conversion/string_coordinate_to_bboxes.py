from comfy_api.latest import io


class StringCoordinateToBBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringCoordinateToBBoxes",
            display_name="String Coordinate to BBoxes",
            category="1hewNodes/conversion",
            inputs=[
                io.String.Input("coordinates_string", default="", multiline=True),
            ],
            outputs=[
                io.Custom("BBOXES").Output(display_name="bboxes"),
            ],
        )

    @classmethod
    def execute(cls, coordinates_string: str) -> io.NodeOutput:
        if not coordinates_string.strip():
            return io.NodeOutput([[]])
            
        # New parsing logic to support JSON-like multi-line formats and flattened lists
        cleaned_string = coordinates_string.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ")
        parts = cleaned_string.split()
        all_coords = []
        for part in parts:
            try:
                all_coords.append(int(float(part)))
            except ValueError:
                continue
        
        bboxes = []
        for i in range(0, len(all_coords), 4):
            if i + 4 <= len(all_coords):
                bboxes.append(all_coords[i:i+4])
                
        if not bboxes:
            return io.NodeOutput([[]])
        sam2_bboxes = [bboxes]
        return io.NodeOutput(sam2_bboxes)
