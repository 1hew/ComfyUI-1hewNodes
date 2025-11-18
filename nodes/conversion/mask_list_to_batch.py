from comfy_api.latest import io
import torch


class MaskListToBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskListToBatch",
            display_name="Mask List to Batch",
            category="1hewNodes/conversion",
            is_input_list=True,
            inputs=[
                io.Mask.Input("mask_list"),
            ],
            outputs=[
                io.Mask.Output(display_name="mask_batch"),
            ],
        )

    @classmethod
    async def execute(cls, mask_list) -> io.NodeOutput:
        if not isinstance(mask_list, (list, tuple)):
            mask_list = [mask_list]

        masks: list[torch.Tensor] = []
        for m in mask_list:
            if isinstance(m, torch.Tensor):
                masks.append(m)

        if not masks:
            return io.NodeOutput(torch.zeros((0, 64, 64)))
        if len(masks) == 1:
            return io.NodeOutput(masks[0])

        max_height = max(mask.shape[-2] for mask in masks)
        max_width = max(mask.shape[-1] for mask in masks)

        padded_masks: list[torch.Tensor] = []
        for mask in masks:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            pad_h = max_height - mask.shape[-2]
            pad_w = max_width - mask.shape[-1]
            if pad_h > 0 or pad_w > 0:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, pad_w, 0, pad_h),
                    mode="constant",
                    value=0,
                )
            padded_masks.append(mask)

        batch_mask = torch.cat(padded_masks, dim=0)
        return io.NodeOutput(batch_mask)
