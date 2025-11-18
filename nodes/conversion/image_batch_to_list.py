from comfy_api.latest import io


class ImageBatchToList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchToList",
            display_name="Image Batch to List",
            category="1hewNodes/conversion",
            inputs=[
                io.Image.Input("image_batch"),
            ],
            outputs=[
                io.Image.Output(display_name="image_list", is_output_list=True),
            ],
        )

    @classmethod
    async def execute(cls, image_batch) -> io.NodeOutput:
        if image_batch is None or image_batch.shape[0] == 0:
            return io.NodeOutput([])
        image_list = [image_batch[i:i+1] for i in range(image_batch.shape[0])]
        return io.NodeOutput(image_list)

