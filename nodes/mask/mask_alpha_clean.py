from __future__ import annotations

from collections import deque

import numpy as np
import torch
from comfy_api.latest import io


class MaskAlphaClean(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskAlphaClean",
            display_name="Mask Alpha Clean",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "clean_strength",
                    options=["soft", "balanced", "strong"],
                    default="balanced",
                ),
                io.Boolean.Input("detect_only", default=False),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="noise_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        clean_strength: str = "balanced",
        detect_only: bool = False,
    ) -> io.NodeOutput:
        if not isinstance(mask, torch.Tensor):
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, empty)

        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.ndim != 3:
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, empty)

        alpha_threshold, threshold_u8, min_area = cls._resolve_preset(clean_strength)

        out_masks: list[torch.Tensor] = []
        out_noises: list[torch.Tensor] = []
        for i in range(int(m.shape[0])):
            cleaned, noise = cls._clean_single_mask(
                alpha=m[i].to(torch.float32),
                alpha_threshold=alpha_threshold,
                threshold_u8=threshold_u8,
                remove_islands_px=min_area,
                detect_only=bool(detect_only),
            )
            out_masks.append(cleaned)
            out_noises.append(noise)

        return io.NodeOutput(
            torch.stack(out_masks, dim=0).to(torch.float32),
            torch.stack(out_noises, dim=0).to(torch.float32),
        )

    @staticmethod
    def _resolve_preset(clean_strength: str) -> tuple[float, int, int]:
        preset = str(clean_strength).strip().lower()
        if preset == "soft":
            return (0.005, 6, 4)
        if preset == "strong":
            return (0.02, 16, 16)
        return (0.01, 10, 8)

    @classmethod
    def _clean_single_mask(
        cls,
        *,
        alpha: torch.Tensor,
        alpha_threshold: float,
        threshold_u8: int,
        remove_islands_px: int,
        detect_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arr = alpha.detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        t_from_u8 = float(threshold_u8) / 255.0
        visible = arr > max(float(alpha_threshold), t_from_u8)
        weak_alpha = (arr > 0.0) & (~visible)
        keep = visible.copy()
        removed = np.zeros_like(visible, dtype=bool)
        if remove_islands_px > 0:
            keep = cls._remove_small_components(visible, min_area=remove_islands_px)
            removed = visible & (~keep)

        noise = weak_alpha | removed
        if detect_only:
            noise_t = torch.from_numpy(noise.astype(np.float32)).to(dtype=alpha.dtype, device=alpha.device)
            return alpha.clamp(0.0, 1.0), noise_t

        out = arr.copy()
        out[~keep] = 0.0
        out[out < float(alpha_threshold)] = 0.0
        out_t = torch.from_numpy(np.clip(out, 0.0, 1.0).astype(np.float32)).to(dtype=alpha.dtype, device=alpha.device)
        noise_t = torch.from_numpy(noise.astype(np.float32)).to(dtype=alpha.dtype, device=alpha.device)
        return out_t, noise_t

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
