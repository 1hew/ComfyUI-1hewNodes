import asyncio
import os
import urllib.request

import cv2
import folder_paths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy_api.latest import io
from PIL import Image


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(self.pool5(hx5))
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6d, hx5), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class BriaRMBG(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.stage1(hxin)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        d1 = _upsample_like(self.side1(hx1d), x)
        d2 = _upsample_like(self.side2(hx2d), x)
        d3 = _upsample_like(self.side3(hx3d), x)
        d4 = _upsample_like(self.side4(hx4d), x)
        d5 = _upsample_like(self.side5(hx5d), x)
        d6 = _upsample_like(self.side6(hx6), x)
        return [torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]


def _bria_preprocess_image(im: Image.Image | np.ndarray, model_input_size: list[int]) -> torch.Tensor:
    if isinstance(im, Image.Image):
        im = np.array(im.convert("RGB"))
    if im.ndim < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear", align_corners=False)
    image = torch.clamp(im_tensor / 255.0, 0.0, 1.0)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return image - mean


def _bria_postprocess_image(result: torch.Tensor, im_size: tuple[int, int] | list[int]) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear", align_corners=False), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    denom = ma - mi
    if torch.abs(denom).item() < 1e-8:
        result = torch.zeros_like(result)
    else:
        result = (result - mi) / denom
    im_array = (result * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    return np.squeeze(im_array)


class DetectRemoveBG(io.ComfyNode):
    MODEL_CACHE = {}

    @classmethod
    def _log_model_ready(
        cls,
        model: str,
        backend: str,
        source: str,
        model_path: str | None = None,
    ) -> None:
        path_value = model_path or "managed-by-runtime"
        cls.log(
            f"ModelReady | model={model} | backend={backend} | source={source} | path={path_value}",
            "info",
        )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DetectRemoveBG",
            display_name="Detect Remove BG",
            category="1hewNodes/detect",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "model",
                    options=[
                        "none",
                        "RMBG-1.4",
                        "RMBG-2.0",
                        "birefnet-general",
                        "birefnet-general-lite",
                        "Inspyrenet",
                    ],
                    default="RMBG-1.4",
                ),
                io.Combo.Input(
                    "add_background",
                    options=["alpha", "white", "black"],
                    default="alpha",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        model: str,
        add_background: str,
    ) -> io.NodeOutput:
        image_batch = []
        mask_batch = []

        model_bundle = cls._prepare_model(model)
        bg_mode = (add_background or "alpha").strip().lower()
        if bg_mode not in {"alpha", "white", "black"}:
            bg_mode = "alpha"
        concurrency = cls._get_inference_concurrency(image=image, model=model, model_bundle=model_bundle)
        sem = asyncio.Semaphore(concurrency)
        tasks = []

        for i in image:
            async def run_one(x=i):
                async with sem:
                    return await asyncio.to_thread(cls._infer_one, x, model, model_bundle, bg_mode)

            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        for img_t, alpha_t in results:
            image_batch.append(img_t)
            mask_batch.append(alpha_t)

        out_image = (
            torch.cat(image_batch, dim=0)
            if image_batch
            else torch.zeros((1, 512, 512, 4 if bg_mode == "alpha" else 3), dtype=torch.float32)
        )
        out_mask = (
            torch.cat(mask_batch, dim=0)
            if mask_batch
            else torch.zeros((1, 512, 512), dtype=torch.float32)
        )
        return io.NodeOutput(out_image, out_mask)

    @classmethod
    def _get_inference_concurrency(cls, image: torch.Tensor, model: str, model_bundle: dict | None) -> int:
        concurrency = max(1, min(len(image), os.cpu_count() or 1))
        if not model_bundle:
            return concurrency

        # Sharing a single CUDA-backed RMBG-1.4 model across worker threads is unstable on the cloud runtime.
        # Keep this backend serial on GPU while preserving parallelism for the lighter backends.
        if model_bundle.get("type") == "rmbg1_4_pth" and str(model_bundle.get("device", "")).lower().startswith("cuda"):
            if concurrency > 1:
                cls.log(
                    f"Using serial inference for model={model} on CUDA to avoid device allocation failures.",
                    "info",
                )
            return 1
        return concurrency

    @classmethod
    def _infer_one(
        cls,
        image_t: torch.Tensor,
        model: str,
        model_bundle: dict | None,
        bg_mode: str,
    ):
        rgb = np.clip(image_t.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            h = rgb.shape[0] if rgb.ndim >= 2 else 512
            w = rgb.shape[1] if rgb.ndim >= 2 else 512
            c = 4 if bg_mode == "alpha" else 3
            return (
                torch.zeros((1, h, w, c), dtype=torch.float32),
                torch.zeros((1, h, w), dtype=torch.float32),
            )

        rgb = rgb[..., :3]
        model_name = (model or "").strip().lower()
        if model_name == "none":
            bg_rgb = cls._estimate_bg_color_bw_priority(rgb)
            alpha = cls._compute_classic_alpha(
                rgb=rgb,
                bg_rgb=bg_rgb,
                tolerance=0.11,
                foreground_protect=0.85,
                edge_softness=1.0,
            )
        else:
            if model_bundle is None:
                raise RuntimeError(
                    f"Model '{model}' could not be loaded. Check the console for [WARNING] messages "
                    "(download failure, missing dependencies, or ONNX/session init errors)."
                )
            alpha = cls._infer_model_alpha(rgb, model, model_bundle)
            if alpha is None:
                raise RuntimeError(f"Model inference returned no alpha for model={model}.")

        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        out_image = cls._compose_output_image(rgb, alpha, bg_mode)
        return torch.from_numpy(out_image).unsqueeze(0), torch.from_numpy(alpha).unsqueeze(0)

    @staticmethod
    def _compose_output_image(rgb: np.ndarray, alpha: np.ndarray, bg_mode: str) -> np.ndarray:
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        if bg_mode == "white":
            out = rgb * a[..., None] + (1.0 - a[..., None]) * 1.0
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        if bg_mode == "black":
            out = rgb * a[..., None]
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), a[..., None]], axis=-1)
        return rgba.astype(np.float32)

    @classmethod
    def _prepare_model(cls, model: str) -> dict | None:
        mode = (model or "").strip()
        if not mode:
            return None
        if mode.lower() == "none":
            cls._log_model_ready(model=mode, backend="classic_only", source="init")
            return {"type": "classic_only"}
        if mode in cls.MODEL_CACHE:
            cached_bundle = cls.MODEL_CACHE[mode]
            cls._log_model_ready(
                model=mode,
                backend=str(cached_bundle.get("type", "unknown")),
                source="cache",
            )
            return cached_bundle

        rembg_dir = cls._get_rembg_dir()
        os.environ["U2NET_HOME"] = rembg_dir

        if mode in {"birefnet-general", "birefnet-general-lite"}:
            try:
                from rembg import new_session
            except Exception:
                cls.log("Missing dependency 'rembg' for BiRefNet models.", "warning")
                return None
            try:
                session = new_session(mode)
            except Exception as e:
                cls.log(f"Create rembg session failed for {mode}: {e}", "warning")
                return None
            bundle = {"type": "rembg_onnx", "session": session}
            cls.MODEL_CACHE[mode] = bundle
            expected_onnx = os.path.join(rembg_dir, f"{mode}.onnx")
            cls._log_model_ready(
                model=mode,
                backend="rembg_onnx",
                source="init",
                model_path=expected_onnx if os.path.isfile(expected_onnx) else None,
            )
            return bundle

        if mode == "RMBG-1.4":
            device = cls._get_worker_device()
            local_pth = cls._resolve_model_file(
                rembg_dir=rembg_dir,
                preferred_name="RMBG-1.4.pth",
                candidate_names=["RMBG-1.4.pth", "model.pth"],
                required_tokens=["rmbg", "1.4"],
            )
            if local_pth is None:
                local_pth = os.path.join(rembg_dir, "RMBG-1.4.pth")
                ok = cls._download_single_model_file(
                    dest_path=local_pth,
                    urls=["https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"],
                    model_label="RMBG-1.4.pth",
                )
                if not ok:
                    return None
            try:
                bundle = cls._create_rmbg1_4_bundle(local_pth=local_pth, device=device)
                cls.MODEL_CACHE[mode] = bundle
                cls._log_model_ready(
                    model=mode,
                    backend=f"rmbg1_4_pth_{bundle['device']}",
                    source="init",
                    model_path=local_pth,
                )
                return bundle
            except Exception as e:
                cls.log(f"Load RMBG-1.4.pth failed: {e}", "warning")
                return None

        if mode == "RMBG-2.0":
            local_onnx = cls._resolve_model_file(
                rembg_dir=rembg_dir,
                preferred_name="RMBG-2.0.onnx",
                candidate_names=["RMBG-2.0.onnx", "model.onnx", "model_fp16.onnx"],
                required_tokens=["rmbg", "2.0"],
            )
            if local_onnx is None:
                local_onnx = os.path.join(rembg_dir, "RMBG-2.0.onnx")
                ok = cls._download_single_model_file(
                    dest_path=local_onnx,
                    urls=[
                        "https://huggingface.co/briaai/RMBG-2.0/resolve/main/onnx/model.onnx",
                        "https://huggingface.co/briaai/RMBG-2.0/resolve/main/onnx/model_fp16.onnx",
                        "https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.onnx",
                        "https://huggingface.co/briaai/RMBG-2.0/resolve/main/RMBG-2.0.onnx",
                    ],
                    model_label="RMBG-2.0.onnx",
                )
                if not ok:
                    return None
            try:
                import onnxruntime as ort
            except Exception as e:
                cls.log("Missing dependency 'onnxruntime' for RMBG-2.0 single-file mode.", "warning")
                raise RuntimeError(
                    "RMBG-2.0 requires the 'onnxruntime' package. "
                    "Install it in the same Python environment as ComfyUI: pip install onnxruntime "
                    "(or onnxruntime-gpu for CUDA; match your CUDA version)."
                ) from e
            providers = ["CPUExecutionProvider"]
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            try:
                session = ort.InferenceSession(local_onnx, sess_options=sess_options, providers=providers)
            except Exception as e:
                cls.log(f"Load RMBG-2.0.onnx failed: {e}", "warning")
                return None
            inputs = session.get_inputs()
            if not inputs:
                return None
            bundle = {
                "type": "rmbg2_onnx",
                "session": session,
                "input_name": inputs[0].name,
                "input_shape": list(inputs[0].shape) if hasattr(inputs[0], "shape") else None,
            }
            cls.MODEL_CACHE[mode] = bundle
            cls._log_model_ready(
                model=mode,
                backend="rmbg2_onnx",
                source="init",
                model_path=local_onnx,
            )
            return bundle

        if mode == "Inspyrenet":
            os.environ["TRANSPARENT_BACKGROUND_FILE_PATH"] = rembg_dir
            try:
                from transparent_background import Remover
            except Exception:
                cls.log("Missing dependency 'transparent-background' for Inspyrenet.", "warning")
                return None
            remover_device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                remover = Remover(jit=False, device=remover_device)
            except Exception as e:
                cls.log(f"Init Inspyrenet failed: {e}", "warning")
                return None
            bundle = {
                "type": "inspyrenet",
                "remover": remover,
                "device": remover_device,
                "remover_kwargs": {"jit": False, "device": remover_device},
            }
            cls.MODEL_CACHE[mode] = bundle
            cls._log_model_ready(
                model=mode,
                backend=f"inspyrenet_{remover_device}",
                source="init",
                model_path=os.path.join(rembg_dir, ".transparent-background"),
            )
            return bundle

        return None

    @classmethod
    def _infer_model_alpha(cls, rgb: np.ndarray, model: str, model_bundle: dict | None) -> np.ndarray | None:
        if model_bundle is None:
            return None
        try:
            if model_bundle.get("type") == "rmbg1_4_pth":
                img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
                try:
                    return cls._run_rmbg1_4_inference(model_bundle, img, rgb.shape[0], rgb.shape[1])
                except Exception as e:
                    current_device = str(model_bundle.get("device", "cpu")).lower()
                    if current_device != "cpu" and cls._is_cuda_oom_error(e):
                        cls.log(
                            "RMBG-1.4 hit CUDA allocation/OOM on the current visible GPU; retrying on CPU.",
                            "warning",
                        )
                        model_bundle = cls._get_rmbg1_4_bundle(model_bundle, device="cpu")
                        alpha = cls._run_rmbg1_4_inference(model_bundle, img, rgb.shape[0], rgb.shape[1])
                        cls.log("RMBG-1.4 CPU fallback succeeded after CUDA allocation/OOM.", "info")
                        return alpha
                    raise

            if model_bundle.get("type") == "rmbg2_onnx":
                session = model_bundle["session"]
                input_name = model_bundle["input_name"]
                input_shape = model_bundle.get("input_shape") or [1, 3, 1024, 1024]
                h_in = input_shape[2] if len(input_shape) >= 4 and isinstance(input_shape[2], int) else 1024
                w_in = input_shape[3] if len(input_shape) >= 4 and isinstance(input_shape[3], int) else 1024
                img_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                img_resized = cv2.resize(img_u8, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                x = img_resized.astype(np.float32) / 255.0
                x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                    [0.229, 0.224, 0.225], dtype=np.float32
                )
                x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
                outputs = session.run(None, {input_name: x})
                if not outputs:
                    return None
                pred = np.asarray(outputs[-1])
                if pred.ndim == 4:
                    pred = pred[0, 0]
                elif pred.ndim == 3:
                    pred = pred[0]
                elif pred.ndim != 2:
                    return None
                if float(np.min(pred)) < 0.0 or float(np.max(pred)) > 1.0:
                    pred = 1.0 / (1.0 + np.exp(-pred))
                alpha = cv2.resize(pred.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                return np.clip(alpha, 0.0, 1.0).astype(np.float32)

            if model_bundle.get("type") == "rembg_onnx":
                from rembg import remove

                session = model_bundle["session"]
                img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
                out = remove(img, session=session, only_mask=True)
                out_np = np.array(out)
                if out_np.ndim == 3:
                    out_np = out_np[..., 0]
                return np.clip(out_np.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

            if model_bundle.get("type") == "inspyrenet":
                remover = model_bundle["remover"]
                img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
                try:
                    out = remover.process(img, type="map")
                except Exception as e:
                    current_device = str(model_bundle.get("device", "cpu")).lower()
                    if current_device != "cpu" and cls._is_cuda_oom_error(e):
                        cls.log(
                            "Inspyrenet hit CUDA OOM on the current visible GPU; retrying on CPU.",
                            "warning",
                        )
                        remover = cls._get_inspyrenet_remover(model_bundle, device="cpu")
                        out = remover.process(img, type="map")
                        cls.log("Inspyrenet CPU fallback succeeded after CUDA OOM.", "info")
                    else:
                        raise
                out_np = np.array(out)
                if out_np.ndim == 3:
                    out_np = out_np[..., 0]
                return np.clip(out_np.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
        except Exception as e:
            cls.log(f"Model inference failed for model={model}: {e}", "warning")
            return None
        return None

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
        return DetectRemoveBG._estimate_bg_color_auto(rgb)

    @staticmethod
    def _compute_classic_alpha(
        rgb: np.ndarray,
        bg_rgb: np.ndarray,
        tolerance: float,
        foreground_protect: float,
        edge_softness: float,
    ) -> np.ndarray:
        diff = np.abs(rgb - bg_rgb.reshape(1, 1, 3)).mean(axis=-1).astype(np.float32)
        b = float(np.percentile(diff, 92))
        span = max(0.02, 0.06 + tolerance * 0.15)
        t0 = np.clip(b - span * (0.35 + 0.25 * foreground_protect), 0.0, 1.0)
        t1 = np.clip(b + span * (0.65 + 0.35 * (1.0 - foreground_protect)), t0 + 1e-4, 1.0)
        x = np.clip((diff - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
        alpha = (x * x * (3.0 - 2.0 * x)).astype(np.float32)
        if edge_softness > 0:
            sigma = 0.4 + 0.5 * float(edge_softness)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha[diff < t0 * 0.9] = 0.0
        alpha[diff > t1 * 1.1] = np.maximum(alpha[diff > t1 * 1.1], 0.98)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _download_single_model_file(cls, dest_path: str, urls: list[str], model_label: str) -> bool:
        tmp_path = f"{dest_path}.part"
        for url in urls:
            try:
                urllib.request.urlretrieve(url, tmp_path)
                os.replace(tmp_path, dest_path)
                cls.log(f"Downloaded {model_label} to models/rembg.", "info")
                return True
            except Exception:
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        cls.log(f"Auto download {model_label} failed from all known URLs.", "warning")
        return False

    @classmethod
    def _get_rembg_dir(cls) -> str:
        path = os.path.join(folder_paths.models_dir, "rembg")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _get_worker_device() -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _is_cuda_oom_error(error: Exception) -> bool:
        message = str(error).lower()
        return any(
            token in message
            for token in (
                "cuda out of memory",
                "out of memory",
                "allocation on device",
                "cuda error: out of memory",
            )
        )

    @classmethod
    def _normalize_rmbg_state_dict(cls, state_dict):
        if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict):
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        return state_dict

    @classmethod
    def _create_rmbg1_4_bundle(cls, local_pth: str, device: str) -> dict:
        model_inst = BriaRMBG()
        state_dict = torch.load(local_pth, map_location=device)
        state_dict = cls._normalize_rmbg_state_dict(state_dict)
        model_inst.load_state_dict(state_dict, strict=True)
        model_inst.to(device).eval()
        return {
            "type": "rmbg1_4_pth",
            "model": model_inst,
            "device": device,
            "preprocess_fn": _bria_preprocess_image,
            "postprocess_fn": _bria_postprocess_image,
            "state_dict_path": local_pth,
        }

    @classmethod
    def _get_rmbg1_4_bundle(cls, model_bundle: dict, device: str) -> dict:
        current_device = str(model_bundle.get("device", "cpu")).lower()
        target_device = str(device).lower()
        if current_device == target_device and model_bundle.get("model") is not None:
            return model_bundle

        if current_device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        bundle = cls._create_rmbg1_4_bundle(
            local_pth=str(model_bundle["state_dict_path"]),
            device=device,
        )
        model_bundle.update(bundle)
        return model_bundle

    @classmethod
    def _run_rmbg1_4_inference(cls, model_bundle: dict, img: Image.Image, height: int, width: int) -> np.ndarray:
        model_inst = model_bundle["model"]
        device = model_bundle["device"]
        preprocess_fn = model_bundle["preprocess_fn"]
        postprocess_fn = model_bundle["postprocess_fn"]
        x = preprocess_fn(img, [1024, 1024]).to(device)
        with torch.no_grad():
            result = model_inst(x)
        pred_u8 = postprocess_fn(result[0][0], (height, width))
        return np.clip(pred_u8.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _get_inspyrenet_remover(cls, model_bundle: dict, device: str):
        if model_bundle.get("device") == device and model_bundle.get("remover") is not None:
            return model_bundle["remover"]

        from transparent_background import Remover

        remover_kwargs = dict(model_bundle.get("remover_kwargs") or {})
        remover_kwargs["device"] = device
        remover = Remover(**remover_kwargs)
        model_bundle["remover"] = remover
        model_bundle["device"] = device
        model_bundle["remover_kwargs"] = remover_kwargs
        return remover

    @classmethod
    def _resolve_model_file(
        cls,
        rembg_dir: str,
        preferred_name: str,
        candidate_names: list[str],
        required_tokens: list[str] | None = None,
    ) -> str | None:
        preferred_lower = preferred_name.lower()
        candidate_lowers = {name.lower() for name in candidate_names}
        required_tokens = [token.lower() for token in (required_tokens or [])]

        preferred_path = os.path.join(rembg_dir, preferred_name)
        if os.path.isfile(preferred_path):
            return preferred_path

        matches: list[tuple[int, str]] = []
        for root, _, files in os.walk(rembg_dir):
            for file_name in files:
                file_name_lower = file_name.lower()
                if file_name_lower not in candidate_lowers:
                    continue

                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, rembg_dir).replace("\\", "/").lower()
                is_root_file = os.path.normcase(root) == os.path.normcase(rembg_dir)
                tokens_ok = all(token in rel_path for token in required_tokens) if required_tokens else True
                if file_name_lower != preferred_lower and not (tokens_ok or is_root_file):
                    continue

                score = 0
                if file_name_lower == preferred_lower:
                    score += 100
                if is_root_file:
                    score += 50
                if tokens_ok:
                    score += 25

                matches.append((score, full_path))

        if not matches:
            return None

        matches.sort(key=lambda item: (-item[0], len(item[1]), item[1].lower()))
        resolved_path = matches[0][1]
        cls.log(
            f"Resolved existing model file | preferred={preferred_name} | path={resolved_path}",
            "info",
        )
        return resolved_path

    @staticmethod
    def log(message: str, level: str = "info"):
        prefix = {
            "info": "[INFO]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
        }.get(level, "[INFO]")
        print(f"# DetectRemoveBG: {prefix} {message}")
