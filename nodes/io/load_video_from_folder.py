from __future__ import annotations
import os
import hashlib
import torch
import numpy as np
import av
from PIL import Image
import torch.nn.functional as F
from comfy_api.latest import io, ui
from server import PromptServer
from aiohttp import web
import shutil

class VideoComponents:
    def __init__(self, images, audio, frame_rate):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate

class VideoFromFile:
    def __init__(self, path):
        self.path = path

    def get_dimensions(self):
        try:
            container = av.open(self.path)
            if len(container.streams.video) > 0:
                stream = container.streams.video[0]
                return stream.width, stream.height
        except Exception:
            pass
        return 0, 0

    def save_to(self, path, format="auto", codec="auto", metadata=None):
        # Since it's already a video file, we can just copy it
        # Note: format/codec conversion is not implemented here for simplicity
        # If format conversion is strictly required, we would need to transcode
        shutil.copy2(self.path, path)

    def get_components(self):
        path = self.path
        try:
            container = av.open(path)
        except Exception as e:
            print(f"Error opening video {path}: {e}")
            return None

        # Video
        frames = []
        video_info = {}
        
        if len(container.streams.video) > 0:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            # Metadata
            try:
                fps = float(stream.average_rate)
            except Exception:
                fps = 0.0
                
            video_info = {
                "source_fps": float(stream.average_rate),
                "source_frame_count": int(stream.frames),
                "source_duration": float(stream.duration * stream.time_base) if stream.duration else 0,
                "fps": fps,
                "width": stream.width,
                "height": stream.height
            }
            
            # Decoding
            try:
                for frame in container.decode(stream):
                    # Convert
                    # We always use RGB for consistency with standard LoadVideo
                    img_np = frame.to_ndarray(format='rgb24')
                    frames.append(img_np)
                    
            except Exception as e:
                print(f"Error decoding video frames {path}: {e}")

        if not frames:
            return None
            
        video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
        
        # Audio
        audio = None
        if len(container.streams.audio) > 0:
            stream = container.streams.audio[0]
            try:
                # Seek to start for audio decoding
                container.seek(0)
                audio_frames = []
                for frame in container.decode(stream):
                    # Convert to float32 planar
                    # PyAV's AudioFrame.to_ndarray does not accept 'format' argument like VideoFrame
                    # It returns numpy array in the frame's native format
                    data = frame.to_ndarray()
                    
                    # Convert to float32 if needed and handle layout
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                        
                    # If format is planar (e.g. fltp), data is [planes, samples]
                    # If format is packed (e.g. flt), data is [samples, channels]
                    # We need [channels, samples]
                    
                    if stream.format.is_planar:
                        # Already [channels, samples] or close to it
                        pass 
                    else:
                        # [samples, channels] -> [channels, samples]
                        if len(data.shape) > 1:
                            data = data.T
                        else:
                             # Mono [samples] -> [1, samples]
                            data = data[None, :]

                    audio_frames.append(data)
                
                if audio_frames:
                    # Concatenate along samples (axis 1)
                    audio_data = np.concatenate(audio_frames, axis=1)
                    
                    # Convert to tensor: [channels, samples]
                    waveform = torch.from_numpy(audio_data).float()
                    
                    # Add batch dimension: [1, channels, samples]
                    waveform = waveform.unsqueeze(0)
                    
                    audio = {
                        "waveform": waveform,
                        "sample_rate": stream.rate
                    }
            except Exception as e:
                print(f"Error decoding audio {path}: {e}")
        
        # Return VideoComponents object instead of dict
        return VideoComponents(
            images=video_tensor,
            audio=audio,
            frame_rate=video_info["fps"]
        )

class LoadVideoFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_LoadVideoFromFolder",
            display_name="Load Video From Folder",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("folder", default=""),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subfolder", default=True),
            ],
            outputs=[
                io.Video.Output(display_name="VIDEO"),
            ],
        )

    @staticmethod
    def get_video_paths(folder, include_subfolder):
        if not os.path.isdir(folder):
            return []
            
        valid_extensions = {'.webm', '.mp4', '.mkv', '.mov', '.avi'}
        video_paths = []

        if include_subfolder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_extensions:
                        video_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isfile(path) and os.path.splitext(file)[1].lower() in valid_extensions:
                    video_paths.append(path)

        # Case-insensitive sort for cross-platform consistency
        video_paths.sort(key=lambda x: x.lower())
        return video_paths

    @classmethod
    def IS_CHANGED(cls, folder, include_subfolder, **kwargs):
        if not os.path.isdir(folder):
            return float("nan")
            
        video_paths = cls.get_video_paths(folder, include_subfolder)
        m = hashlib.sha256()
        for path in video_paths:
            try:
                mtime = os.path.getmtime(path)
                m.update(f"{path}:{mtime}".encode())
            except OSError:
                continue
                
        return m.hexdigest()

    @classmethod
    async def execute(cls, folder: str, index: int, include_subfolder: bool) -> io.NodeOutput:
        video_paths = cls.get_video_paths(folder, include_subfolder)
        count = len(video_paths)

        if count == 0:
            return io.NodeOutput(None)

        idx = index % count
        path = video_paths[idx]
        
        return io.NodeOutput(VideoFromFile(path))


@PromptServer.instance.routes.get("/1hew/view_video_from_folder")
async def view_video_from_folder(request):
    folder = request.query.get("folder")
    index_str = request.query.get("index", "0")
    include_subfolder = request.query.get("include_subfolder", "true").lower() == "true"
    
    if not folder or not os.path.isdir(folder):
        return web.Response(status=404)
        
    try:
        index = int(index_str)
    except ValueError:
        index = 0

    video_paths = LoadVideoFromFolder.get_video_paths(folder, include_subfolder)
    if not video_paths:
        return web.Response(status=404)

    idx = index % len(video_paths)
    path = video_paths[idx]
    
    return web.FileResponse(path)

