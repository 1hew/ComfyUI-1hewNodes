from comfy_api.latest import io


class MaskBatchToList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskBatchToList",
            display_name="Mask Batch to List",
            category="1hewNodes/conversion",
            inputs=[
                io.Mask.Input("mask_batch"),
            ],
            outputs=[
                io.Mask.Output(display_name="mask_list", is_output_list=True),
            ],
        )

    @classmethod
    async def execute(cls, mask_batch) -> io.NodeOutput:
        if mask_batch is None or mask_batch.shape[0] == 0:
            return io.NodeOutput([])
        mask_list = [mask_batch[i:i+1] for i in range(mask_batch.shape[0])]
        return io.NodeOutput(mask_list)

