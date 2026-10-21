import cv2
import numpy as np
import torch

from comfy_api.latest import io

from ...utils._image_ops import (
    apply_levels_numpy,
)

class ImageHLFreqCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageHLFreqCombine",
            display_name="Image HL Freq Combine",
            category="1hewNodes/image/hlfreq",
            inputs=[
                io.Image.Input("high_freq"),
                io.Image.Input("low_freq"),
                io.Combo.Input("method", options=["rgb", "hsv", "igbi"], default="rgb"),
                io.Float.Input("high_strength", default=1.0, min=0.0, max=2.0),
                io.Float.Input("low_strength", default=1.0, min=0.0, max=2.0),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        high_freq: torch.Tensor,
        low_freq: torch.Tensor,
        method: str,
        high_strength: float,
        low_strength: float,
    ) -> io.NodeOutput:
        hb = int(high_freq.shape[0])
        lb = int(low_freq.shape[0])
        bsz = max(hb, lb)

        high_t = cls._align_batch(high_freq, bsz)
        low_t = cls._align_batch(low_freq, bsz)

        if method in ("rgb", "hsv"):
            high_t = (high_t - 0.5) * float(high_strength) + 0.5
        else:
            high_t = high_t * float(high_strength)
        low_t = low_t * float(low_strength)

        high_t = torch.clamp(high_t, 0.0, 1.0)
        low_t = torch.clamp(low_t, 0.0, 1.0)

        high_np = high_t.detach().cpu().numpy()
        low_np = low_t.detach().cpu().numpy()

        out_list: list[torch.Tensor] = []
        for i in range(bsz):
            h_img = high_np[i]
            l_img = low_np[i]
            if method == "rgb":
                res = cls._recombine_linear_light(h_img, l_img)
            elif method == "hsv":
                res = cls._recombine_hsv(h_img, l_img)
            else:
                res = cls._recombine_igbi(h_img, l_img)
            out_list.append(torch.from_numpy(res.astype(np.float32)))

        return io.NodeOutput(torch.stack(out_list))

    @staticmethod
    def _align_batch(t: torch.Tensor, target: int) -> torch.Tensor:
        b = int(t.shape[0])
        if b == target:
            return t
        rep = target // b
        rem = target % b
        parts = [t.repeat(rep, 1, 1, 1)]
        if rem > 0:
            parts.append(t[:rem])
        return torch.cat(parts, dim=0)

    @staticmethod
    def _recombine_linear_light(high: np.ndarray, low: np.ndarray):
        res = 2.0 * high + low - 1.0
        return np.clip(res, 0.0, 1.0)

    @staticmethod
    def _recombine_hsv(high: np.ndarray, low: np.ndarray):
        low_hsv = cv2.cvtColor((low * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_low = cv2.split(low_hsv)
        v_low_n = v_low.astype(np.float32) / 255.0
        high_v = high[:, :, 0]
        v_comb = np.clip((2.0 * v_low_n + high_v - 1.0) * 255.0, 0, 255).astype(np.uint8)
        comb_hsv = cv2.merge([h_ch, s_ch, v_comb])
        comb_rgb = cv2.cvtColor(comb_hsv, cv2.COLOR_HSV2RGB)
        return comb_rgb.astype(np.float32) / 255.0


    @classmethod
    def _recombine_igbi(cls, high: np.ndarray, low: np.ndarray):
        hu8 = (high * 255.0).astype(np.uint8)
        lu8 = (low * 255.0).astype(np.uint8)
        mixed = (hu8.astype(np.float32) * 0.65 + lu8.astype(np.float32) * 0.35).astype(np.uint8)
        out = apply_levels_numpy(mixed, 83, 172, 1.0, 0, 255)
        return out.astype(np.float32) / 255.0

