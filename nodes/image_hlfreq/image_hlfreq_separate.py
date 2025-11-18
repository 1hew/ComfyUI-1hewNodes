import cv2
import numpy as np
import torch

from comfy_api.latest import io

BLUR_COEFFICIENT = 1


class ImageHLFreqSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageHLFreqSeparate",
            display_name="Image HL Freq Separate",
            category="1hewNodes/image/hlfreq",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["rgb", "hsv", "igbi"], default="rgb"),
                io.Float.Input("blur_radius", default=10.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="high_freq"),
                io.Image.Output(display_name="low_freq"),
                io.Image.Output(display_name="combine"),
            ],
        )

    @classmethod
    async def execute(
        cls, image: torch.Tensor, method: str, blur_radius: float
    ) -> io.NodeOutput:
        bsz = int(image.shape[0])
        eff_radius = float(blur_radius) * float(BLUR_COEFFICIENT)

        img_np = image.detach().cpu().numpy() if image.is_cuda else image.detach().numpy()

        highs: list[torch.Tensor] = []
        lows: list[torch.Tensor] = []
        combs: list[torch.Tensor] = []

        for i in range(bsz):
            img = img_np[i]

            if method == "igbi":
                h, l = cls._igbi_separation(img, eff_radius)
                c = cls._recombine_igbi(h, l)
            elif method == "hsv":
                h, l = cls._hsv_separation(img, eff_radius)
                c = cls._recombine_hsv(h, l)
            else:
                h, l = cls._rgb_separation(img, eff_radius)
                c = cls._recombine_linear_light(h, l)

            highs.append(torch.from_numpy(h.astype(np.float32)))
            lows.append(torch.from_numpy(l.astype(np.float32)))
            combs.append(torch.from_numpy(c.astype(np.float32)))

        return io.NodeOutput(torch.stack(highs), torch.stack(lows), torch.stack(combs))

    @staticmethod
    def _ensure_odd_radius(radius: float) -> int:
        r = int(radius)
        if r % 2 == 0:
            r += 1
        return max(r, 3)

    @classmethod
    def _igbi_separation(cls, image: np.ndarray, blur_radius: float):
        r = cls._ensure_odd_radius(blur_radius)
        u8 = (image * 255.0).astype(np.uint8)
        low = cv2.GaussianBlur(u8, (r, r), 0).astype(np.float32) / 255.0
        inv = 255 - u8
        blur = cv2.GaussianBlur(u8, (r, r), 0)
        mix = (inv.astype(np.float32) * 0.5 + blur.astype(np.float32) * 0.5).astype(np.uint8)
        high = (255 - mix).astype(np.float32) / 255.0
        return high, low

    @classmethod
    def _recombine_igbi(cls, high: np.ndarray, low: np.ndarray):
        hu8 = (high * 255.0).astype(np.uint8)
        lu8 = (low * 255.0).astype(np.uint8)
        mixed = (hu8.astype(np.float32) * 0.65 + lu8.astype(np.float32) * 0.35).astype(np.uint8)
        out = cls._apply_levels_numpy(mixed, 83, 172, 1.0, 0, 255)
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _rgb_separation(image: np.ndarray, blur_radius: float):
        r = ImageHLFreqSeparate._ensure_odd_radius(blur_radius)
        low = cv2.GaussianBlur(image.astype(np.float32), (r, r), 0)
        gray = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur_g = cv2.GaussianBlur(gray, (r, r), 0)
        high_g = (gray.astype(np.float32) - blur_g.astype(np.float32)) / 255.0 + 0.5
        high_g = np.clip(high_g, 0.0, 1.0)
        high = np.stack([high_g] * 3, axis=-1)
        return high.astype(np.float32), low.astype(np.float32)

    @staticmethod
    def _hsv_separation(image: np.ndarray, blur_radius: float):
        r = ImageHLFreqSeparate._ensure_odd_radius(blur_radius)
        hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v_blur = cv2.GaussianBlur(v, (r, r), 0)
        high_v = (v.astype(np.float32) - v_blur.astype(np.float32)) / 255.0 + 0.5
        high_v = np.clip(high_v, 0.0, 1.0)
        high = np.stack([high_v] * 3, axis=-1)
        hsv_low = hsv.copy()
        hsv_low[..., 2] = v_blur
        low = cv2.cvtColor(hsv_low, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return high.astype(np.float32), low.astype(np.float32)

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

    @staticmethod
    def _apply_levels_numpy(
        image_array: np.ndarray,
        black_point: int,
        white_point: int,
        gray_point: float = 1.0,
        output_black_point: int = 0,
        output_white_point: int = 255,
    ) -> np.ndarray:
        bp = int(max(0, min(255, black_point)))
        wp = int(max(0, min(255, white_point)))
        obp = int(max(0, min(255, output_black_point)))
        owp = int(max(0, min(255, output_white_point)))

        result = np.zeros_like(image_array)
        for i in range(image_array.shape[2]):
            ch = image_array[:, :, i].astype(np.float32)
            ch = np.clip(ch, bp, wp)
            denom = max(1, (wp - bp))
            ch = (ch - bp) / float(denom) * 255.0
            if gray_point != 1.0:
                ch = 255.0 * (ch / 255.0) ** (1.0 / float(gray_point))
            ch = (ch / 255.0) * (owp - obp) + obp
            result[:, :, i] = np.clip(ch, 0, 255).astype(np.uint8)
        return result
