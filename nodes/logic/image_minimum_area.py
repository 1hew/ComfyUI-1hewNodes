import math

from comfy_api.latest import io
from PIL import Image
import torch

from ..image_resize.image_resize_universal import ImageResizeUniversal


class ImageMinimumArea(io.ComfyNode):
    """Upscale an IMAGE until its area reaches a minimum square reference area.

    - ``length_to_sq_area`` defines the reference square side length (target
      area = N²).
    - ``fit`` behaves like ``Image Resize Universal``: crop / pad / stretch.
    - ``divisible_by`` rounds both final dimensions up to a multiple.
    - ``resize_bool`` reports whether an actual upscale happened.
    """

    METHODS = ["nearest", "bilinear", "lanczos", "bicubic", "hamming", "box"]
    FIT_MODES = ["crop", "pad", "stretch"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageMinimumArea",
            display_name="Image Minimum Area",
            category="1hewNodes/logic",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Int.Input(
                    "length_to_sq_area",
                    default=1024,
                    min=1,
                    max=65536,
                    step=1,
                ),
                io.Combo.Input("method", options=cls.METHODS, default="lanczos"),
                io.Int.Input("divisible_by", default=1, min=1, max=1024, step=1),
                io.Combo.Input("fit", options=cls.FIT_MODES, default="crop"),
                io.String.Input("pad_color", default="1.0"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Boolean.Output(display_name="resize_bool"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        length_to_sq_area: int,
        method: str,
        divisible_by: int,
        fit: str,
        pad_color: str,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if image is None or image.ndim != 4:
            raise ValueError("image must be a BHWC IMAGE tensor")

        height, width = int(image.shape[1]), int(image.shape[2])
        if height < 1 or width < 1:
            raise ValueError("image dimensions must be positive")

        target_area = int(length_to_sq_area) ** 2
        current_area = width * height

        # Area already meets the target: pass everything through unchanged.
        if current_area >= target_area:
            out_mask = cls._passthrough_mask(mask, width, height)
            return io.NodeOutput(image, out_mask, False)

        target_width, target_height = cls._target_size(
            width, height, target_area, int(divisible_by)
        )
        sampler = cls._resize_sampler(method)

        out_image = cls._resize_image(
            image, target_width, target_height, fit, sampler, pad_color, width, height
        )
        out_mask = cls._resize_mask(
            mask, target_width, target_height, fit, sampler, width, height, len(image)
        )

        return io.NodeOutput(out_image, out_mask, True)

    @staticmethod
    def _target_size(
        width: int,
        height: int,
        target_area: int,
        divisible_by: int,
    ) -> tuple[int, int]:
        """Scale both sides by one factor, then round each side up to a multiple.

        Mirrors ``Image Resize Universal``'s ``length_to_sq_area`` calculation:
        a square reference of N means a target area of N².  ``divisible_by`` may
        add a few pixels so both final dimensions satisfy a model size constraint.
        """
        scale = math.sqrt(target_area / float(width * height))
        target_width = max(1, int(math.ceil(width * scale)))
        target_height = max(1, int(math.ceil(height * scale)))

        if divisible_by > 1:
            target_width = ((target_width + divisible_by - 1) // divisible_by) * divisible_by
            target_height = ((target_height + divisible_by - 1) // divisible_by) * divisible_by

        return target_width, target_height

    @staticmethod
    def _resize_sampler(method: str):
        if method == "bicubic":
            return Image.BICUBIC
        if method == "hamming":
            return Image.HAMMING
        if method == "bilinear":
            return Image.BILINEAR
        if method == "box":
            return Image.BOX
        if method == "nearest":
            return Image.NEAREST
        return Image.LANCZOS

    @classmethod
    def _resize_image(
        cls,
        image: torch.Tensor,
        target_width: int,
        target_height: int,
        fit: str,
        sampler,
        pad_color: str,
        orig_width: int,
        orig_height: int,
    ) -> torch.Tensor:
        """Resize every image frame with the requested fit mode, keeping alpha."""
        out = []
        for img in image:
            frame = torch.unsqueeze(img, 0)
            pil = ImageResizeUniversal.tensor2pil(frame)

            if "A" in pil.getbands() or "transparency" in pil.info:
                rgba = pil.convert("RGBA")
                rgb = rgba.convert("RGB")
                alpha = rgba.getchannel("A")

                resized_rgb = ImageResizeUniversal.fit_resize_image(
                    rgb, target_width, target_height, fit, sampler, pad_color
                ).convert("RGB")
                resized_alpha = ImageResizeUniversal.fit_resize_mask(
                    alpha, target_width, target_height, fit, sampler, orig_width, orig_height
                ).convert("L")

                resized_rgb.putalpha(resized_alpha)
                out.append(ImageResizeUniversal.pil2tensor(resized_rgb))
            else:
                resized = ImageResizeUniversal.fit_resize_image(
                    pil.convert("RGB"), target_width, target_height, fit, sampler, pad_color
                )
                out.append(ImageResizeUniversal.pil2tensor(resized))

        return torch.cat(out, dim=0)

    @classmethod
    def _resize_mask(
        cls,
        mask: torch.Tensor | None,
        target_width: int,
        target_height: int,
        fit: str,
        sampler,
        orig_width: int,
        orig_height: int,
        image_count: int,
    ) -> torch.Tensor:
        """Resize an optional input mask, or synthesize a default mask."""
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            out = []
            for m in mask:
                frame = torch.unsqueeze(m, 0)
                pil = ImageResizeUniversal.tensor2pil(frame).convert("L")
                resized = ImageResizeUniversal.fit_resize_mask(
                    pil, target_width, target_height, fit, sampler, orig_width, orig_height
                )
                out.append(ImageResizeUniversal.image2mask(resized))
            return torch.cat(out, dim=0)

        out = []
        for _ in range(image_count):
            default = cls._default_mask(target_width, target_height, fit, orig_width, orig_height)
            out.append(ImageResizeUniversal.image2mask(default))
        return torch.cat(out, dim=0)

    @staticmethod
    def _default_mask(
        target_width: int,
        target_height: int,
        fit: str,
        orig_width: int,
        orig_height: int,
    ) -> Image.Image:
        """Build a default mask matching the requested fit mode.

        - pad: target-sized mask (content white, padding black).
        - crop: source-sized mask marking the region kept after the crop
          (matches ``Image Resize Universal`` / ``Jimeng``).
        - stretch: target-sized all-white mask.
        """
        if fit in ("pad", "crop"):
            return ImageResizeUniversal.generate_default_mask(
                target_width, target_height, fit, orig_width, orig_height
            )
        return Image.new("L", (target_width, target_height), 255)

    @staticmethod
    def _passthrough_mask(
        mask: torch.Tensor | None,
        width: int,
        height: int,
    ) -> torch.Tensor:
        if mask is not None:
            return mask
        return ImageResizeUniversal.image2mask(Image.new("L", (width, height), 255))
