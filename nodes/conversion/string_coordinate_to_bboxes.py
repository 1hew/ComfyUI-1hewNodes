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
        lines = coordinates_string.strip().split("\n")
        bboxes = []
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
                bboxes.append(coords[:4])
        if not bboxes:
            return io.NodeOutput([[]])
        sam2_bboxes = [bboxes]
        return io.NodeOutput(sam2_bboxes)
