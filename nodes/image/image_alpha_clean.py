from __future__ import annotations

from collections import deque

import numpy as np
import torch
from comfy_api.latest import io


class ImageAlphaClean(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlphaClean",
            display_name="Image Alpha Clean",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "clean_strength",
                    options=["soft", "balanced", "strong"],
                    default="balanced",
                ),
                io.Boolean.Input("detect_only", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="noise_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        clean_strength: str = "balanced",
        detect_only: bool = False,
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            empty = torch.zeros((0, 64, 64, 4), dtype=torch.float32)
            empty_mask = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, empty_mask)

        img = image
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            empty = torch.zeros((0, 64, 64, 4), dtype=torch.float32)
            empty_mask = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, empty_mask)

        alpha_threshold, threshold_u8_v, min_area, bleed = cls._resolve_preset(clean_strength)

        out_images: list[torch.Tensor] = []
        out_noise_masks: list[torch.Tensor] = []

        for i in range(int(img.shape[0])):
            frame = img[i].to(torch.float32)
            rgba = cls._to_rgba(frame)

            cleaned, noise_mask = cls._clean_single_rgba(
                rgba=rgba,
                alpha_threshold=alpha_threshold,
                threshold_u8=threshold_u8_v,
                remove_islands_px=min_area,
                keep_largest_component=False,
                bleed_radius=bleed,
                zero_rgb_when_alpha_zero=True,
                detect_only=bool(detect_only),
            )
            out_images.append(cleaned)
            out_noise_masks.append(noise_mask.clamp(0.0, 1.0))

        image_out = torch.stack(out_images, dim=0).to(torch.float32)
        noise_out = torch.stack(out_noise_masks, dim=0).to(torch.float32)
        return io.NodeOutput(image_out, noise_out)

    @classmethod
    def _clean_single_rgba(
        cls,
        rgba: torch.Tensor,
        *,
        alpha_threshold: float,
        threshold_u8: int,
        remove_islands_px: int,
        keep_largest_component: bool,
        bleed_radius: int,
        zero_rgb_when_alpha_zero: bool,
        detect_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arr = rgba.detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3].copy()

        t_from_u8 = float(threshold_u8) / 255.0
        visible = alpha > max(float(alpha_threshold), t_from_u8)
        weak_alpha = (alpha > 0.0) & (~visible)
        keep = visible.copy()
        removed = np.zeros_like(visible, dtype=bool)
        if remove_islands_px > 0:
            keep = cls._remove_small_components(visible, min_area=remove_islands_px)
            removed = visible & (~keep)
        if keep_largest_component:
            largest = cls._keep_largest_component(keep)
            removed = removed | (keep & (~largest))
            keep = largest

        noise = weak_alpha | removed
        if detect_only:
            noise_t = torch.from_numpy(noise.astype(np.float32)).to(dtype=rgba.dtype, device=rgba.device)
            return rgba.clamp(0.0, 1.0), noise_t

        alpha_clean = alpha.copy()
        alpha_clean[~keep] = 0.0
        alpha_clean[alpha_clean < float(alpha_threshold)] = 0.0

        rgb_clean = rgb.copy()
        if bleed_radius > 0:
            rgb_clean = cls._bleed_rgb_into_transparent(rgb_clean, alpha_clean > 0.0, bleed_radius)
        if zero_rgb_when_alpha_zero:
            rgb_clean[alpha_clean <= 0.0] = 0.0

        out = np.concatenate(
            [np.clip(rgb_clean, 0.0, 1.0), np.clip(alpha_clean, 0.0, 1.0)[:, :, None]],
            axis=2,
        ).astype(np.float32)
        out_t = torch.from_numpy(out).to(dtype=rgba.dtype, device=rgba.device)
        noise_t = torch.from_numpy(noise.astype(np.float32)).to(dtype=rgba.dtype, device=rgba.device)
        return out_t, noise_t

    @staticmethod
    def _resolve_preset(clean_strength: str) -> tuple[float, int, int, int]:
        preset = str(clean_strength).strip().lower()
        if preset == "soft":
            return (0.005, 6, 4, 1)
        if preset == "strong":
            return (0.02, 16, 16, 2)
        return (0.01, 10, 8, 2)

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 1:
            return mask
        h, w = int(mask.shape[0]), int(mask.shape[1])
        visited = np.zeros((h, w), dtype=bool)
        keep = np.zeros((h, w), dtype=bool)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            q: deque[tuple[int, int]] = deque()
            q.append((y0, x0))
            comp: list[tuple[int, int]] = [(y0, x0)]

            while q:
                y, x = q.popleft()
                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or (not mask[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))

            if len(comp) >= int(min_area):
                for cy, cx in comp:
                    keep[cy, cx] = True
        return keep

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        h, w = int(mask.shape[0]), int(mask.shape[1])
        visited = np.zeros((h, w), dtype=bool)
        out = np.zeros((h, w), dtype=bool)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        best_comp: list[tuple[int, int]] = []
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            q: deque[tuple[int, int]] = deque()
            q.append((y0, x0))
            comp: list[tuple[int, int]] = [(y0, x0)]
            while q:
                y, x = q.popleft()
                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or (not mask[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
            if len(comp) > len(best_comp):
                best_comp = comp

        for y, x in best_comp:
            out[y, x] = True
        return out

    @staticmethod
    def _bleed_rgb_into_transparent(rgb: np.ndarray, known_mask: np.ndarray, radius: int) -> np.ndarray:
        h, w = int(known_mask.shape[0]), int(known_mask.shape[1])
        if h <= 0 or w <= 0 or radius <= 0:
            return rgb
        known = known_mask.copy()
        unknown = ~known
        if not np.any(known) or not np.any(unknown):
            return rgb

        out = rgb.copy()
        for _ in range(int(radius)):
            if not np.any(unknown):
                break
            sum_rgb = np.zeros_like(out, dtype=np.float32)
            cnt = np.zeros((h, w), dtype=np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n_known = ImageAlphaClean._shift_bool(known, dy, dx)
                    if not np.any(n_known):
                        continue
                    n_rgb = ImageAlphaClean._shift_rgb(out, dy, dx)
                    m = unknown & n_known
                    if not np.any(m):
                        continue
                    sum_rgb[m] += n_rgb[m]
                    cnt[m] += 1.0
            fill = unknown & (cnt > 0.0)
            if not np.any(fill):
                break
            out[fill] = sum_rgb[fill] / cnt[fill, None]
            known[fill] = True
            unknown = ~known
        return out

    @staticmethod
    def _shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        h, w = int(mask.shape[0]), int(mask.shape[1])
        out = np.zeros((h, w), dtype=bool)

        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy) if dy >= 0 else h
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx) if dx >= 0 else w

        dst_y0 = max(0, dy)
        dst_x0 = max(0, dx)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _shift_rgb(rgb: np.ndarray, dy: int, dx: int) -> np.ndarray:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        out = np.zeros((h, w, 3), dtype=np.float32)

        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy) if dy >= 0 else h
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx) if dx >= 0 else w

        dst_y0 = max(0, dy)
        dst_x0 = max(0, dx)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1, :] = rgb[src_y0:src_y1, src_x0:src_x1, :]
        return out

    @staticmethod
    def _to_rgba(frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 3:
            return torch.zeros((64, 64, 4), dtype=torch.float32, device=frame.device)
        h, w, c = int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])
        if c >= 4:
            return frame[:, :, :4].to(torch.float32).clamp(0.0, 1.0)
        if c == 3:
            alpha = torch.ones((h, w, 1), dtype=frame.dtype, device=frame.device)
            return torch.cat([frame[:, :, :3], alpha], dim=2).to(torch.float32).clamp(0.0, 1.0)
        if c == 1:
            rgb = frame.repeat(1, 1, 3)
            alpha = torch.ones((h, w, 1), dtype=frame.dtype, device=frame.device)
            return torch.cat([rgb, alpha], dim=2).to(torch.float32).clamp(0.0, 1.0)
        return torch.zeros((h, w, 4), dtype=torch.float32, device=frame.device)
