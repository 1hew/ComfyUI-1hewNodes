import comfy.utils
import inspect
import math
import node_helpers


class TextEncodeQwenImageEditKeepSize:
    @classmethod
    def INPUT_TYPES(cls):
        # 默认提供一个下划线风格的首端口 image_1；前端扩展追加更多 image_X
        opt_inputs = {
            "vae": ("VAE",),
            "image_1": ("IMAGE", {"forceInput": True}),
        }

        try:
            stack = inspect.stack()
            if len(stack) > 2 and stack[2].function == "get_input_info":
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        k = str(key)
                        if k.startswith("image_"):
                            return "IMAGE", {"forceInput": True}
                        if k == "vae":
                            return "VAE", {}
                        return "IMAGE", {"forceInput": True}

                opt_inputs = AllContainer()
        except Exception:
            pass

        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "keep_size": (["none", "first", "all"], {"default": "first"}),
                "base_size": (["1024", "1536", "2048"], {"default": "1024"}),
            },
            "optional": opt_inputs,
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "1hewNodes/condition"

    def encode(
        self,
        clip,
        prompt,
        vae=None,
        keep_size="all",
        base_size="1024",
        **kwargs,
    ):
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
        def _iter_images(kws):
            keys = []
            for k in kws.keys():
                if k.startswith("image_"):
                    suffix = k[len("image_") :]
                    if suffix.isdigit():
                        keys.append((int(suffix), k))
            keys.sort(key=lambda x: x[0])
            for _, key in keys:
                yield kws.get(key)

        for i, image in enumerate(_iter_images(kwargs)):
            if image is None:
                continue
            samples = image.movedim(-1, 1)
            in_h = int(samples.shape[2])
            in_w = int(samples.shape[3])
            # 固定视觉语义输入尺寸为 384x384，以完全对齐原生节点行为
            vl_w = 384
            vl_h = 384
            s_vl = comfy.utils.common_upscale(
                samples, vl_w, vl_h, "area", "disabled"
            )
            images_vl.append(s_vl.movedim(1, -1))

            if vae is not None:
                if keep_size == "all" or (keep_size == "first" and i == 0):
                    if (in_w % 8 == 0) and (in_h % 8 == 0):
                        ref_latents.append(vae.encode(image[:, :, :, :3]))
                    else:
                        # 非 8 倍数时按 none 模式的面积缩放逻辑处理
                        base = int(base_size)
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
                    base = int(base_size)
                    total = int(base * base)
                    scale_by = math.sqrt(total / (in_w * in_h))
                    vae_w = round(in_w * scale_by / 8.0) * 8
                    vae_h = round(in_h * scale_by / 8.0) * 8
                    s_vae = comfy.utils.common_upscale(
                        samples, vae_w, vae_h, "area", "disabled"
                    )
                    ref_latents.append(
                        vae.encode(
                            s_vae.movedim(1, -1)[:, :, :, :3]
                        )
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
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "1hew_TextEncodeQwenImageEditKeepSize": TextEncodeQwenImageEditKeepSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_TextEncodeQwenImageEditKeepSize": "Text Encode QwenImageEdit Keep Size",
}