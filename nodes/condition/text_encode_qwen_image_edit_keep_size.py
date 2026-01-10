import comfy.utils
import math
import node_helpers
import torch
import torch.nn.functional as F
from comfy_api.latest import io


class TextEncodeQwenImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [
            io.Custom("CLIP").Input("clip"),
            io.Custom("VAE").Input("vae", optional=True),
            io.Custom("IMAGE").Input("image_1", optional=True),
            io.String.Input("prompt", default="", multiline=True),
            io.Combo.Input("reference_skip_prep", options=["none", "first", "all"], default="first"),
            io.Int.Input("reference_sq_area", default=1024, min=64, max=8192, step=16),
            io.Combo.Input("vision_embed", options=["crop", "pad", "stretch", "area"], default="stretch"),
        ]
        return io.Schema(
            node_id="1hew_TextEncodeQwenImageEdit",
            display_name="Text Encode QwenImageEdit",
            category="1hewNodes/condition",
            inputs=inputs,
            outputs=[io.Custom("CONDITIONING").Output(display_name="conditioning")],
        )

    @classmethod
    async def execute(
        cls,
        clip,
        prompt,
        vae=None,
        reference_skip_prep="first",
        vision_embed="area",
        reference_sq_area=1024,
        **kwargs,
    ) -> io.NodeOutput:
        images_vl = []
        ref_latents = []

        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, "
            "size, texture, objects, background), then explain how the "
            "user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while "
            "maintaining consistency with the original input where "
            "appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

        image_prompt = ""

        # 收集动态图像输入，仅支持 image_1、image_2、...（下划线风格）
        ordered = []
        for k in kwargs.keys():
            if k.startswith("image_"):
                suf = k[len("image_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        for i, key in enumerate([k for _, k in ordered]):
            image = kwargs.get(key)
            if image is None:
                continue
            samples = image.movedim(-1, 1)
            in_h = int(samples.shape[2])
            in_w = int(samples.shape[3])
            if vision_embed == "area":
                total = int(384 * 384)
                scale_by = math.sqrt(total / (in_w * in_h))
                vl_w = max(1, int(round(in_w * scale_by)))
                vl_h = max(1, int(round(in_h * scale_by)))
                s_vl = comfy.utils.common_upscale(
                    samples, vl_w, vl_h, "area", "disabled"
                )
            else:
                vl_w = 384
                vl_h = 384
                if vision_embed == "crop":
                    s_vl = comfy.utils.common_upscale(
                        samples, vl_w, vl_h, "area", "center"
                    )
                elif vision_embed == "pad":
                    scale_by = min(vl_w / in_w, vl_h / in_h)
                    resized_w = max(1, int(round(in_w * scale_by)))
                    resized_h = max(1, int(round(in_h * scale_by)))
                    resized = comfy.utils.common_upscale(
                        samples, resized_w, resized_h, "area", "disabled"
                    )
                    pad_left = (vl_w - resized_w) // 2
                    pad_right = vl_w - resized_w - pad_left
                    pad_top = (vl_h - resized_h) // 2
                    pad_bottom = vl_h - resized_h - pad_top
                    s_vl = F.pad(
                        resized,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0.0,
                    )
                else:
                    s_vl = comfy.utils.common_upscale(
                        samples, vl_w, vl_h, "area", "disabled"
                    )
            images_vl.append(s_vl.movedim(1, -1))

            if vae is not None:
                keep = reference_skip_prep
                if keep == "all" or (keep == "first" and i == 0):
                    if (in_w % 8 == 0) and (in_h % 8 == 0):
                        ref_latents.append(vae.encode(image[:, :, :, :3]))
                    else:
                        base = int(reference_sq_area)
                        total = int(base * base)
                        scale_by = math.sqrt(total / (in_w * in_h))
                        vae_w = round(in_w * scale_by / 8.0) * 8
                        vae_h = round(in_h * scale_by / 8.0) * 8
                        s_vae = comfy.utils.common_upscale(
                            samples, vae_w, vae_h, "area", "disabled"
                        )
                        ref_latents.append(
                            vae.encode(s_vae.movedim(1, -1)[:, :, :, :3])
                        )
                else:
                    base = int(reference_sq_area)
                    total = int(base * base)
                    scale_by = math.sqrt(total / (in_w * in_h))
                    vae_w = round(in_w * scale_by / 8.0) * 8
                    vae_h = round(in_h * scale_by / 8.0) * 8
                    s_vae = comfy.utils.common_upscale(
                        samples, vae_w, vae_h, "area", "disabled"
                    )
                    ref_latents.append(
                        vae.encode(s_vae.movedim(1, -1)[:, :, :, :3])
                    )

            image_prompt += (
                f"Picture {i + 1}: "
                "<|vision_start|><|image_pad|><|vision_end|>"
            )

        tokens = clip.tokenize(
            image_prompt + prompt, images=images_vl, llama_template=llama_template
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return io.NodeOutput(conditioning)
