import asyncio
import logging
import requests
from io import BytesIO
from typing import Optional

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest import io



class URLToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_URLToVideo",
            display_name="URL to Video",
            category="1hewNodes/conversion",
            inputs=[
                io.String.Input("video_url"),
                io.Int.Input("timeout", default=30, min=5, max=300, step=1),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    async def execute(cls, video_url: str, timeout: int) -> io.NodeOutput:
        if not video_url or not video_url.strip():
            raise ValueError("视频URL不能为空")

        if not video_url.startswith(("http://", "https://")):
            raise ValueError("无效的URL格式，必须以http://或https://开头")

        try:
            video_data = await cls.download_video_from_url(video_url, timeout)
            if video_data is None:
                raise RuntimeError("视频下载失败")

            video_object = VideoFromFile(video_data)
            return io.NodeOutput(video_object)
        except Exception as e:
            logging.error(f"URL转换视频失败: {str(e)}")
            raise

    @classmethod
    async def download_video_from_url(
        cls, video_url: str, timeout: int = 30
    ) -> Optional[BytesIO]:
        def _fetch() -> Optional[BytesIO]:
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
                response = requests.get(
                    video_url, headers=headers, timeout=timeout, stream=True
                )
                response.raise_for_status()

                buf = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        buf.write(chunk)
                buf.seek(0)
                return buf
            except Exception as exc:
                logging.error(f"下载视频失败: {str(exc)}")
                return None

        return await asyncio.to_thread(_fetch)
