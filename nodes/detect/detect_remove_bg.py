import asyncio
import importlib.util
import os
import urllib.request

import cv2
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

_BRIA_MODULE_CACHE = None


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
        concurrency = max(1, min(len(image), os.cpu_count() or 1))
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            local_pth = os.path.join(rembg_dir, "RMBG-1.4.pth")
            if not os.path.isfile(local_pth):
                ok = cls._download_single_model_file(
                    dest_path=local_pth,
                    urls=["https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"],
                    model_label="RMBG-1.4.pth",
                )
                if not ok:
                    return None
            try:
                bria_module = cls._load_bria_module_from_easyuse()
                model_inst = bria_module.BriaRMBG()
                state_dict = torch.load(local_pth, map_location=device)
                if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
                    state_dict = state_dict["state_dict"]
                if isinstance(state_dict, dict):
                    if any(k.startswith("module.") for k in state_dict.keys()):
                        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
                model_inst.load_state_dict(state_dict, strict=True)
                model_inst.to(device).eval()
                bundle = {
                    "type": "rmbg1_4_pth",
                    "model": model_inst,
                    "device": device,
                    "preprocess_fn": bria_module.preprocess_image,
                    "postprocess_fn": bria_module.postprocess_image,
                }
                cls.MODEL_CACHE[mode] = bundle
                cls._log_model_ready(
                    model=mode,
                    backend="rmbg1_4_pth",
                    source="init",
                    model_path=local_pth,
                )
                return bundle
            except Exception as e:
                cls.log(f"Load RMBG-1.4.pth failed: {e}", "warning")
                return None

        if mode == "RMBG-2.0":
            local_onnx = os.path.join(rembg_dir, "RMBG-2.0.onnx")
            if not os.path.isfile(local_onnx):
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
            except Exception:
                cls.log("Missing dependency 'onnxruntime' for RMBG-2.0 single-file mode.", "warning")
                return None
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
            try:
                remover = Remover(jit=False)
            except Exception as e:
                cls.log(f"Init Inspyrenet failed: {e}", "warning")
                return None
            bundle = {"type": "inspyrenet", "remover": remover}
            cls.MODEL_CACHE[mode] = bundle
            cls._log_model_ready(
                model=mode,
                backend="inspyrenet",
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
                model_inst = model_bundle["model"]
                device = model_bundle["device"]
                preprocess_fn = model_bundle["preprocess_fn"]
                postprocess_fn = model_bundle["postprocess_fn"]
                img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
                x = preprocess_fn(img, [1024, 1024]).to(device)
                with torch.no_grad():
                    result = model_inst(x)
                pred_u8 = postprocess_fn(result[0][0], (rgb.shape[0], rgb.shape[1]))
                return np.clip(pred_u8.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

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
                out = remover.process(img, type="map")
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

    @classmethod
    def _load_bria_module_from_easyuse(cls):
        global _BRIA_MODULE_CACHE
        if _BRIA_MODULE_CACHE is not None:
            return _BRIA_MODULE_CACHE
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(custom_nodes_dir, "ComfyUI-Easy-Use", "py", "modules", "briaai", "rembg.py"),
            os.path.join(custom_nodes_dir, "comfyui-easy-use", "py", "modules", "briaai", "rembg.py"),
        ]
        base_path = getattr(folder_paths, "base_path", None)
        if base_path:
            candidates.extend(
                [
                    os.path.join(base_path, "custom_nodes", "ComfyUI-Easy-Use", "py", "modules", "briaai", "rembg.py"),
                    os.path.join(base_path, "custom_nodes", "comfyui-easy-use", "py", "modules", "briaai", "rembg.py"),
                ]
            )
        module_path = next((p for p in candidates if os.path.isfile(p)), None)
        if not module_path:
            raise FileNotFoundError("Cannot find ComfyUI-Easy-Use briaai/rembg.py")
        spec = importlib.util.spec_from_file_location("detect_remove_bg_easyuse_bria", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key in ("BriaRMBG", "preprocess_image", "postprocess_image"):
            if not hasattr(module, key):
                raise AttributeError(f"Missing '{key}' in {module_path}")
        _BRIA_MODULE_CACHE = module
        return _BRIA_MODULE_CACHE

    @staticmethod
    def log(message: str, level: str = "info"):
        prefix = {
            "info": "[INFO]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
        }.get(level, "[INFO]")
        print(f"# DetectRemoveBG: {prefix} {message}")
