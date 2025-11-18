from __future__ import annotations
import av
import asyncio
import folder_paths
import json
import math
import os
import random
import numpy as np
import torch
from fractions import Fraction
from PIL import Image
from typing import Optional
from comfy_api.latest import io, ui


class SaveVideoRGBA(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveVideoRGBA",
            display_name="Save Video RGBA",
            category="1hewNodes/save",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=120.0, step=1.0
                ),
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip=(
                        "The prefix for the file to save. This may include "
                        "formatting information such as %date:yyyy-MM-dd% "
                        "or %Empty Latent Image.width% to include values "
                        "from nodes."
                    ),
                ),
                io.Boolean.Input("only_preview", default=False),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls, images, fps, filename_prefix, only_preview, audio=None, **kwargs
    ) -> io.NodeOutput:
        b, h, w, c = images.shape
        has_alpha = c == 4

        divisible_by = 2
        if w % divisible_by != 0 or h % divisible_by != 0:
            new_w = w - (w % divisible_by)
            new_h = h - (h % divisible_by)
            print(f"Resize video from {w}x{h} to {new_w}x{new_h}")

            async def _resize_one(i):
                def _do():
                    img = Image.fromarray(
                        (images[i].cpu().numpy() * 255).astype(np.uint8)
                    )
                    resized = img.resize((new_w, new_h), Image.LANCZOS)
                    arr = np.array(resized).astype(np.float32) / 255.0
                    return torch.from_numpy(arr).unsqueeze(0)

                return await asyncio.to_thread(_do)

            tasks = [_resize_one(i) for i in range(b)]
            parts = await asyncio.gather(*tasks)
            images = torch.cat(parts, dim=0)
            b, h, w, c = images.shape

        results = []

        if only_preview or has_alpha:
            temp_dir = folder_paths.get_temp_directory()
            suffix = "".join(
                random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
            )
            prefix_append = f"ComfyUI_temp_{suffix}"
            (
                full_dir,
                base,
                counter,
                subfolder,
                filename_prefix,
            ) = folder_paths.get_save_image_path(prefix_append, temp_dir, w, h)
            fmt = "webm" if has_alpha else "auto"
            ext = cls._get_extension(fmt)
            file = f"{base}_{counter:05}_.{ext}"
            await asyncio.to_thread(
                cls._save_video,
                images,
                audio,
                float(fps),
                os.path.join(full_dir, file),
                fmt,
                "auto",
                None,
            )
            results.append(ui.SavedResult(file, subfolder, io.FolderType.temp))

        if not only_preview:
            out_dir = folder_paths.get_output_directory()
            (
                full_dir,
                base,
                counter,
                subfolder,
                filename_prefix,
            ) = folder_paths.get_save_image_path(filename_prefix, out_dir, w, h)
            fmt = "mov" if has_alpha else "auto"
            codec = "prores_ks" if has_alpha else "auto"
            ext = cls._get_extension(fmt)
            file = f"{base}_{counter:05}_.{ext}"
            await asyncio.to_thread(
                cls._save_video,
                images,
                audio,
                float(fps),
                os.path.join(full_dir, file),
                fmt,
                codec,
                None,
            )
            if not has_alpha:
                results.append(ui.SavedResult(file, subfolder, io.FolderType.output))

        return io.NodeOutput(ui=ui.PreviewVideo(results))

    @staticmethod
    def _get_extension(fmt: str) -> str:
        if fmt == "auto":
            return "mp4"
        if fmt == "mp4":
            return "mp4"
        if fmt == "webm":
            return "webm"
        if fmt == "mov":
            return "mov"
        return "mp4"

    @staticmethod
    def _save_video(
        images,
        audio: Optional[dict],
        fps: float,
        path: str,
        fmt: str,
        codec: str,
        metadata: Optional[dict],
    ) -> None:
        has_alpha = images.shape[-1] == 4 if len(images.shape) == 4 else False

        if has_alpha:
            if fmt == "auto":
                fmt = "webm"
            if codec == "auto":
                codec = "libvpx-vp9"
            if fmt not in ["webm", "mov"]:
                raise ValueError("Unsupported alpha format")
        else:
            if fmt != "auto" and fmt not in ["mp4", "webm", "mov"]:
                raise ValueError("Unsupported format")
            if codec == "auto":
                codec = "h264"
            elif codec not in ["h264", "libvpx-vp9", "prores_ks"]:
                raise ValueError("Unsupported codec")

        options = {}
        fmt_str = None if fmt == "auto" else fmt
        if fmt in ["mp4", "mov"]:
            options["movflags"] = "use_metadata_tags"

        with av.open(path, mode="w", format=fmt_str, options=options) as output:
            if metadata is not None:
                for k, v in metadata.items():
                    output.metadata[k] = json.dumps(v)

            rate = Fraction(round(fps * 1000), 1000)
            vstream = output.add_stream(codec, rate=rate)
            vstream.width = images.shape[2]
            vstream.height = images.shape[1]

            if has_alpha:
                if codec in ["libvpx-vp9"]:
                    vstream.pix_fmt = "yuva420p"
                elif codec in ["prores_ks"]:
                    vstream.pix_fmt = "yuva444p10le"
                else:
                    vstream.pix_fmt = "yuva420p"
            else:
                vstream.pix_fmt = "yuv420p"

            astream = None
            asr = 1
            if audio:
                asr = int(audio["sample_rate"])
                acodec = "libopus" if fmt == "webm" else "aac"
                astream = output.add_stream(acodec, rate=asr)

            for i, frame in enumerate(images):
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy()
                if has_alpha:
                    vf = av.VideoFrame.from_ndarray(img, format="rgba")
                    vf = vf.reformat(format=vstream.pix_fmt)
                else:
                    vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                    vf = vf.reformat(format=vstream.pix_fmt)
                pkt = vstream.encode(vf)
                output.mux(pkt)

            pkt = vstream.encode(None)
            output.mux(pkt)

            if astream and audio:
                waveform = audio["waveform"]
                frames = images.shape[0]
                total = math.ceil((asr / rate) * frames)
                waveform = waveform[:, :, :total]
                arr = waveform.movedim(2, 1).reshape(1, -1).float().numpy()
                af = av.AudioFrame.from_ndarray(
                    arr,
                    format="flt",
                    layout="mono" if waveform.shape[1] == 1 else "stereo",
                )
                af.sample_rate = asr
                af.pts = 0
                output.mux(astream.encode(af))
                output.mux(astream.encode(None))