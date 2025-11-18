from comfy_api.latest import io
import torch


class ImageListToBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageListToBatch",
            display_name="Image List to Batch",
            category="1hewNodes/conversion",
            is_input_list=True,
            inputs=[
                io.Image.Input("image_list"),
            ],
            outputs=[
                io.Image.Output(display_name="image_batch"),
            ],
        )

    @classmethod
    async def execute(cls, image_list) -> io.NodeOutput:
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list]

        images: list[torch.Tensor] = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                images.append(img)

        if not images:
            return io.NodeOutput(torch.zeros((0, 64, 64, 3)))
        if len(images) == 1:
            return io.NodeOutput(images[0])

        max_height = max(img.shape[-3] for img in images)
        max_width = max(img.shape[-2] for img in images)

        padded_images: list[torch.Tensor] = []
        for img in images:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            pad_h = max_height - img.shape[-3]
            pad_w = max_width - img.shape[-2]
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(
                    img,
                    (0, 0, 0, pad_w, 0, pad_h),
                    mode="constant",
                    value=0,
                )
            padded_images.append(img)

        batch_image = torch.cat(padded_images, dim=0)
        return io.NodeOutput(batch_image)
