import asyncio
import os
from comfy_api.latest import io, ui
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import torch

class DetectGuideLine(io.ComfyNode):
    """引导线检测"""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DetectGuideLine",
            display_name="Detect Guide Line",
            category="1hewNodes/detect",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("canny_low", default=0.2, min=0.0, max=1.0, step=0.01),
                io.Float.Input("canny_high", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seg_min_len", default=40, min=1, max=300, step=1),
                io.Int.Input("seg_max_gap", default=8, min=1, max=100, step=1),
                io.Float.Input("guide_filter", default=0.6, min=0.1, max=1.0, step=0.1),
                io.Int.Input("guide_width", default=2, min=1, max=100, step=1),
                io.Int.Input("cluster_eps", default=30, min=1, max=100, step=5),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="line_image"),
                io.Mask.Output(display_name="line_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        canny_low: float,
        canny_high: float,
        seg_min_len: int,
        seg_max_gap: int,
        guide_filter: float,
        guide_width: int,
        cluster_eps: int,
    ) -> io.NodeOutput:
        batch_images = []
        batch_lines_only = []
        batch_line_masks = []

        concurrency = max(1, min(len(image), os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for i in image:
            async def run_one(x=i):
                async with sem:
                    return await asyncio.to_thread(
                        cls._process_image,
                        x,
                        canny_low,
                        canny_high,
                        seg_min_len,
                        seg_max_gap,
                        guide_filter,
                        guide_width,
                        cluster_eps,
                    )
            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        for img_t, lines_t, mask_t in results:
            batch_images.append(img_t)
            batch_lines_only.append(lines_t)
            batch_line_masks.append(mask_t)

        out_image = (
            torch.cat(batch_images, dim=0)
            if batch_images
            else torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        )
        out_lines_only = (
            torch.cat(batch_lines_only, dim=0)
            if batch_lines_only
            else torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        )
        out_line_mask = (
            torch.cat(batch_line_masks, dim=0)
            if batch_line_masks
            else torch.zeros((1, 512, 512), dtype=torch.float32)
        )

        return io.NodeOutput(out_image, out_lines_only, out_line_mask)

    @classmethod
    def _process_image(
        cls,
        i: torch.Tensor,
        canny_low: float,
        canny_high: float,
        seg_min_len: int,
        seg_max_gap: int,
        guide_filter: float,
        guide_width: int,
        cluster_eps: int,
    ):
        img_np = (i.cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_rgb = img_np.copy()
        h, w = img_rgb.shape[:2]
        lines_only = np.zeros((h, w, 3), dtype=np.uint8)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        red = (255, 0, 0)
        default_vp = (w // 2, h // 2)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        low = int(np.clip(float(canny_low), 0.0, 1.0) * 255)
        high = int(np.clip(float(canny_high), 0.0, 1.0) * 255)
        high = max(high, low)
        edges = cv2.Canny(gray_blur, low, high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=seg_min_len,
            maxLineGap=seg_max_gap,
        )

        if lines is None:
            return (
                torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0),
            )

        intersections = []
        for m in range(len(lines)):
            x1, y1, x2, y2 = lines[m][0]
            dx, dy = x2 - x1, y2 - y1
            x1_ext = x1 - 0.1 * dx
            y1_ext = y1 - 0.1 * dy
            x2_ext = x2 + 0.1 * dx
            y2_ext = y2 + 0.1 * dy
            for n in range(m + 1, len(lines)):
                x3, y3, x4, y4 = lines[n][0]
                dx2, dy2 = x4 - x3, y4 - y3
                x3_ext = x3 - 0.1 * dx2
                y3_ext = y3 - 0.1 * dy2
                x4_ext = x4 + 0.1 * dx2
                y4_ext = y4 + 0.1 * dy2
                den = (x1_ext - x2_ext) * (y3_ext - y4_ext) - (
                    y1_ext - y2_ext
                ) * (x3_ext - x4_ext)
                if den != 0:
                    t_num = (x1_ext - x3_ext) * (y3_ext - y4_ext) - (
                        y1_ext - y3_ext
                    ) * (x3_ext - x4_ext)
                    u_num = -(
                        (x1_ext - x2_ext) * (y1_ext - y3_ext)
                        - (y1_ext - y2_ext) * (x1_ext - x3_ext)
                    )
                    t = t_num / den
                    x = x1_ext + t * (x2_ext - x1_ext)
                    y = y1_ext + t * (y2_ext - y1_ext)
                    if (-w <= x <= 2 * w) and (-h <= y <= 2 * h):
                        intersections.append((x, y))

        if not intersections:
            return (
                torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0),
            )

        intersections_np = np.array(intersections)
        try:
            dbscan = DBSCAN(eps=cluster_eps, min_samples=5)
            clusters = dbscan.fit_predict(intersections_np)
            valid_clusters = clusters[clusters != -1]
            if valid_clusters.size == 0:
                vanishing_point = default_vp
            else:
                counts = np.bincount(valid_clusters)
                largest_cluster_idx = int(np.argmax(counts))
                cluster_points = intersections_np[clusters == largest_cluster_idx]
                if cluster_points.size == 0:
                    vanishing_point = default_vp
                else:
                    vanishing_point = (
                        int(np.mean(cluster_points[:, 0])),
                        int(np.mean(cluster_points[:, 1])),
                    )
        except Exception:
            vanishing_point = default_vp

        line_scores = []
        vx, vy = vanishing_point
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dir1 = np.array([x1 - vx, y1 - vy])
            dir2 = np.array([x2 - vx, y2 - vy])
            line_dir = np.array([x2 - x1, y2 - y1])
            norm = np.linalg.norm(line_dir)
            if norm < 1e-8:
                line_scores.append(0.0)
                continue
            line_dir_norm = line_dir / norm
            n1 = np.linalg.norm(dir1)
            n2 = np.linalg.norm(dir2)
            s1 = (
                np.abs(np.dot(dir1 / (n1 + 1e-8), line_dir_norm)) if n1 > 1e-8 else 0.0
            )
            s2 = (
                np.abs(np.dot(dir2 / (n2 + 1e-8), line_dir_norm)) if n2 > 1e-8 else 0.0
            )
            line_scores.append(max(s1, s2))

        if len(line_scores) == 0:
            filtered_lines = []
        else:
            th = np.percentile(line_scores, 100 - (guide_filter * 60))
            filtered_lines = [
                line for idx, line in enumerate(lines) if line_scores[idx] >= th
            ]

        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_rgb, (x1, y1), vanishing_point, red, guide_width)
            cv2.line(img_rgb, (x2, y2), vanishing_point, red, guide_width)
            cv2.line(lines_only, (x1, y1), vanishing_point, red, guide_width)
            cv2.line(lines_only, (x2, y2), vanishing_point, red, guide_width)
            cv2.line(line_mask, (x1, y1), vanishing_point, 255, guide_width)

        cv2.circle(img_rgb, vanishing_point, max(5, guide_width), red, -1)
        cv2.circle(lines_only, vanishing_point, max(5, guide_width), red, -1)
        cv2.circle(line_mask, vanishing_point, max(5, guide_width), 255, -1)

        return (
            torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0),
        )
