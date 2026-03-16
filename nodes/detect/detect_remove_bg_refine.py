import asyncio
import os

import cv2
import numpy as np
import torch
from comfy_api.latest import io


class DetectRemoveBGRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DetectRemoveBGRefine",
            display_name="Detect Remove BG Refine",
            category="1hewNodes/detect",
            inputs=[
                io.Image.Input("image"),  # 必须是原始 image
                io.Mask.Input("mask"),  # RMBG 模型输出 mask
                io.Combo.Input("type", options=["bitmap", "vector"], default="bitmap"),
                io.Float.Input("subject_protect", default=0.85, min=0.0, max=1.0, step=0.05),
                io.Float.Input("feather", default=1.0, min=0.0, max=64.0, step=0.5),
                io.Float.Input("decolor_edge", default=1.0, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output(display_name="image"),  # RGBA
                io.Mask.Output(display_name="mask"),  # refined alpha
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        type: str,
        subject_protect: float,
        feather: float,
        decolor_edge: float,
    ) -> io.NodeOutput:
        rgba_batch = []
        alpha_batch = []

        subject_protect = float(np.clip(subject_protect, 0.0, 1.0))
        feather = float(np.clip(feather, 0.0, 64.0))
        decolor_edge = float(np.clip(decolor_edge, 0.0, 1.0))
        edge_type = (type or "bitmap").strip().lower()

        n_img = int(image.shape[0]) if image.ndim >= 4 else 0
        n_mask = int(mask.shape[0]) if mask.ndim >= 3 else 0
        n = max(n_img, n_mask, 1)

        concurrency = max(1, min(n, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []

        for idx in range(n):
            img_idx = min(idx, max(n_img - 1, 0))
            mask_idx = min(idx, max(n_mask - 1, 0))
            img_t = image[img_idx] if n_img > 0 else torch.zeros((512, 512, 3), dtype=torch.float32)
            m_t = mask[mask_idx] if n_mask > 0 else torch.zeros((512, 512), dtype=torch.float32)

            async def run_one(x=img_t, m=m_t):
                async with sem:
                    return await asyncio.to_thread(
                        cls._refine_one,
                        x,
                        m,
                        edge_type,
                        subject_protect,
                        feather,
                        decolor_edge,
                    )

            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        for rgba_t, alpha_t in results:
            rgba_batch.append(rgba_t)
            alpha_batch.append(alpha_t)

        out_rgba = (
            torch.cat(rgba_batch, dim=0)
            if rgba_batch
            else torch.zeros((1, 512, 512, 4), dtype=torch.float32)
        )
        out_alpha = (
            torch.cat(alpha_batch, dim=0)
            if alpha_batch
            else torch.zeros((1, 512, 512), dtype=torch.float32)
        )
        return io.NodeOutput(out_rgba, out_alpha)

    @classmethod
    def _refine_one(
        cls,
        image_t: torch.Tensor,
        mask_t: torch.Tensor,
        edge_type: str,
        subject_protect: float,
        feather: float,
        decolor_edge: float,
    ):
        # 输入 image 必须是原图 RGB（不是模型处理后的 image）
        rgb = np.clip(image_t.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if rgb.ndim != 3:
            h = rgb.shape[0] if rgb.ndim >= 2 else 512
            w = rgb.shape[1] if rgb.ndim >= 2 else 512
            rgb = np.zeros((h, w, 3), dtype=np.float32)
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        elif rgb.shape[-1] < 3:
            h, w = rgb.shape[:2]
            pad = np.zeros((h, w, 3), dtype=np.float32)
            pad[..., : rgb.shape[-1]] = rgb
            rgb = pad

        alpha = np.clip(mask_t.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        if alpha.ndim != 2:
            alpha = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        if alpha.shape[0] != rgb.shape[0] or alpha.shape[1] != rgb.shape[1]:
            alpha = np.clip(
                cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR),
                0.0,
                1.0,
            ).astype(np.float32)

        h, w = rgb.shape[:2]
        edge_softness = cls._feather_to_internal_softness(feather, h, w)
        alpha = cls._refine_model_alpha(alpha, edge_softness, subject_protect)
        if edge_type == "vector":
            alpha = cls._apply_hard_shape_edges(alpha, edge_softness, subject_protect)

        # 固定白/黑优先背景估计 + 去色边
        bg_rgb = cls._estimate_bg_color_bw_priority(rgb)
        rgb_clean = cls._decontaminate(rgb, alpha, bg_rgb, decolor_edge)
        # 不做 premultiply，直接输出普通 RGBA
        rgba = np.concatenate([rgb_clean, alpha[..., None]], axis=-1).astype(np.float32)
        return torch.from_numpy(rgba).unsqueeze(0), torch.from_numpy(alpha).unsqueeze(0)

    @staticmethod
    def _feather_to_internal_softness(feather: float, h: int, w: int) -> float:
        min_dim = float(max(1, min(int(h), int(w))))
        px = float(np.clip(feather, 0.0, 64.0))
        softness = 0.35 + (px / max(min_dim * 0.08, 1.0)) * 5.0
        return float(np.clip(softness, 0.0, 5.0))

    @staticmethod
    def _estimate_bg_color_auto(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = int(np.clip(min(h, w) // 12, 2, min(h, w)))
        border = np.concatenate(
            [
                rgb[:s, :, :].reshape(-1, 3),
                rgb[-s:, :, :].reshape(-1, 3),
                rgb[:, :s, :].reshape(-1, 3),
                rgb[:, -s:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        return np.median(border, axis=0).astype(np.float32)

    @staticmethod
    def _estimate_bg_color_bw_priority(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = int(np.clip(min(h, w) // 12, 2, min(h, w)))
        border = np.concatenate(
            [
                rgb[:s, :, :].reshape(-1, 3),
                rgb[-s:, :, :].reshape(-1, 3),
                rgb[:, :s, :].reshape(-1, 3),
                rgb[:, -s:, :].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        gray = np.mean(border, axis=1)
        low_var = float(np.std(gray)) <= 0.06
        mean_gray = float(np.mean(gray))
        if low_var and mean_gray >= 0.92:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if low_var and mean_gray <= 0.08:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return DetectRemoveBGRefine._estimate_bg_color_auto(rgb)

    @staticmethod
    def _apply_edge_antialias(alpha: np.ndarray, edge_softness: float) -> np.ndarray:
        alpha_in = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        fg = alpha_in >= 0.5
        if fg.size == 0 or np.all(fg) or not np.any(fg):
            return alpha_in
        fg_u8 = fg.astype(np.uint8)
        bg_u8 = (1 - fg_u8).astype(np.uint8)
        dist_in = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(bg_u8, cv2.DIST_L2, 3)
        sdf = dist_out - dist_in
        aa_width = 1.1 + 0.45 * float(np.clip(edge_softness, 0.0, 5.0))
        aa_alpha = np.clip(0.5 - sdf / max(aa_width, 1e-6), 0.0, 1.0).astype(np.float32)
        edge_band = np.abs(sdf) <= (aa_width * 1.6 + 0.8)
        if not np.any(edge_band):
            return alpha_in
        blend_w = np.clip(0.52 + 0.06 * float(np.clip(edge_softness, 0.0, 5.0)), 0.52, 0.82)
        out = alpha_in.copy()
        out[edge_band] = (1.0 - blend_w) * out[edge_band] + blend_w * aa_alpha[edge_band]
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _refine_model_alpha(
        cls,
        alpha: np.ndarray,
        edge_softness: float,
        foreground_protect: float,
    ) -> np.ndarray:
        alpha_in = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        edge_band = (alpha_in > 0.02) & (alpha_in < 0.98)
        alpha_out = cv2.bilateralFilter(
            alpha_in,
            d=0,
            sigmaColor=0.06 + 0.05 * float(edge_softness),
            sigmaSpace=1.2 + 2.2 * float(edge_softness),
        )
        if edge_softness > 0:
            alpha_out = cv2.GaussianBlur(
                alpha_out,
                (0, 0),
                sigmaX=0.35 * float(edge_softness),
                sigmaY=0.35 * float(edge_softness),
            )
        core_th = 0.78 + 0.18 * float(np.clip(foreground_protect, 0.0, 1.0))
        core = alpha_in >= core_th
        alpha_out[core] = np.maximum(alpha_out[core], 0.99)
        alpha_out[edge_band] = 0.65 * alpha_out[edge_band] + 0.35 * alpha_in[edge_band]
        alpha_out = cls._apply_edge_antialias(alpha_out, edge_softness)
        alpha_out[core] = np.maximum(alpha_out[core], 0.99)
        return np.clip(alpha_out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _chaikin_closed_curve(points: np.ndarray, iterations: int = 1) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return pts
        for _ in range(max(0, int(iterations))):
            n = pts.shape[0]
            out = np.empty((n * 2, 2), dtype=np.float32)
            nxt = np.roll(pts, -1, axis=0)
            out[0::2] = 0.75 * pts + 0.25 * nxt
            out[1::2] = 0.25 * pts + 0.75 * nxt
            pts = out
        return pts

    @classmethod
    def _apply_hard_shape_edges(
        cls,
        alpha: np.ndarray,
        edge_softness: float,
        foreground_protect: float,
    ) -> np.ndarray:
        alpha_in = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        fg = alpha_in >= 0.5
        if fg.size == 0 or np.all(fg) or not np.any(fg):
            return alpha_in
        h, w = alpha_in.shape
        min_dim = min(h, w)
        ss = 8 if min_dim <= 512 else (4 if min_dim <= 1536 else 2)
        up_w = max(2, int(w * ss))
        up_h = max(2, int(h * ss))
        alpha_up = cv2.resize(alpha_in, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        alpha_up_u8 = (alpha_up * 255.0).astype(np.uint8)
        _, fg_up = cv2.threshold(alpha_up_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_up_50 = (alpha_up_u8 >= 127).astype(np.uint8) * 255
        fg_up = cv2.bitwise_or(fg_up, fg_up_50)
        k = max(1, ss // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        fg_up = cv2.morphologyEx(fg_up, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_up = cv2.morphologyEx(fg_up, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(fg_up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or hierarchy is None:
            return alpha_in
        out_up = np.zeros((up_h, up_w), dtype=np.float32)
        hierarchy = hierarchy[0]
        # 固定最小面积策略（不暴露参数）
        min_area = max(8, (up_h * up_w) // 20000)
        smooth_iter = 2 if ss >= 4 else 1
        for i, cnt in enumerate(contours):
            area = abs(cv2.contourArea(cnt))
            if area < min_area:
                continue
            pts = cnt[:, 0, :].astype(np.float32)
            if pts.shape[0] >= 8:
                pts = cls._chaikin_closed_curve(pts, iterations=smooth_iter)
            cnt_draw = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            parent = hierarchy[i][3]
            color = 1.0 if parent < 0 else 0.0
            cv2.drawContours(out_up, [cnt_draw], -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        edge_soft = float(np.clip(edge_softness, 0.0, 5.0))
        if edge_soft > 0:
            sigma = (0.04 + 0.06 * edge_soft) * ss
            out_up = cv2.GaussianBlur(out_up, (0, 0), sigmaX=sigma, sigmaY=sigma)
        core_th = 0.78 + 0.18 * float(np.clip(foreground_protect, 0.0, 1.0))
        core_up = alpha_up >= core_th
        out_up[core_up] = np.maximum(out_up[core_up], 0.997)
        out = cv2.resize(out_up, (w, h), interpolation=cv2.INTER_AREA)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _decontaminate(
        rgb: np.ndarray,
        alpha: np.ndarray,
        bg_rgb: np.ndarray,
        despill: float,
    ) -> np.ndarray:
        despill = float(np.clip(despill, 0.0, 1.0))
        if despill <= 0:
            return np.clip(rgb, 0.0, 1.0).astype(np.float32)
        a = np.clip(alpha[..., None], 1e-5, 1.0)
        bg = bg_rgb.reshape(1, 1, 3).astype(np.float32)
        recovered = (rgb - (1.0 - a) * bg) / a
        recovered = np.clip(recovered, 0.0, 1.0)
        w = (np.sqrt(a) * despill).astype(np.float32)
        out = rgb * (1.0 - w) + recovered * w
        return np.clip(out, 0.0, 1.0).astype(np.float32)
