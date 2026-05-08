import asyncio
import re

from comfy_api.latest import io
import torch


class ImageBatchIndex(io.ComfyNode):
    _BRACKET_RANGE_RE = re.compile(
        r"(?P<open>[\[\(])\s*(?P<start>-?\d+)\s*,\s*(?P<end>-?\d+)\s*(?P<close>[\]\)])\s*(?:\:\s*(?P<step>\d+))?"
    )
    _DASH_RANGE_RE = re.compile(
        r"(?P<start>-?\d+)\s*-\s*(?P<end>-?\d+)\s*(?:\:\s*(?P<step>\d+))?"
    )
    _INT_RE = re.compile(r"(?P<value>-?\d+)")

    @staticmethod
    def _empty_image(image: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (0,) + tuple(image.shape[1:]),
            dtype=image.dtype,
            device=image.device,
        )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchIndex",
            display_name="Image Batch Index",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("index", default="0", multiline=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, index: str) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])
            if batch_size <= 0:
                return io.NodeOutput(cls._empty_image(image))

            indices = await asyncio.to_thread(cls._parse_indices, index or "", batch_size)
            valid = [i for i in indices if 0 <= i < batch_size]

            if not valid:
                return io.NodeOutput(cls._empty_image(image))

            chunk_size = 512
            chunks = [valid[i : i + chunk_size] for i in range(0, len(valid), chunk_size)]

            async def _gather_chunk(chunk: list[int]) -> torch.Tensor:
                return await asyncio.to_thread(lambda: image[chunk])

            parts = await asyncio.gather(*[_gather_chunk(ch) for ch in chunks])
            extracted = torch.cat(parts, dim=0)
            return io.NodeOutput(extracted)
        except Exception:
            return io.NodeOutput(cls._empty_image(image))

    @classmethod
    def _parse_indices(cls, text: str, batch_size: int) -> list[int]:
        normalized = cls._normalize_text(text)
        indices: list[int] = []
        pos = 0
        length = len(normalized)

        while pos < length:
            ch = normalized[pos]
            if ch.isspace() or ch in ",;":
                pos += 1
                continue

            match = cls._BRACKET_RANGE_RE.match(normalized, pos)
            if match:
                indices.extend(
                    cls._expand_bracket_range(
                        match.group("open"),
                        int(match.group("start")),
                        int(match.group("end")),
                        match.group("close"),
                        int(match.group("step")) if match.group("step") else 1,
                        batch_size,
                    )
                )
                pos = match.end()
                continue

            match = cls._DASH_RANGE_RE.match(normalized, pos)
            if match:
                indices.extend(
                    cls._expand_dash_range(
                        int(match.group("start")),
                        int(match.group("end")),
                        int(match.group("step")) if match.group("step") else 1,
                        batch_size,
                    )
                )
                pos = match.end()
                continue

            match = cls._INT_RE.match(normalized, pos)
            if match:
                indices.append(cls._resolve_index(int(match.group("value")), batch_size))
                pos = match.end()
                continue

            pos += 1

        return indices

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (
            text.replace("【", "[")
            .replace("】", "]")
            .replace("（", "(")
            .replace("）", ")")
            .replace("，", ",")
            .replace("：", ":")
            .replace("；", ",")
        )

    @staticmethod
    def _resolve_index(value: int, batch_size: int) -> int:
        if value < 0:
            return batch_size + value
        return value

    @classmethod
    def _expand_bracket_range(
        cls,
        open_bracket: str,
        start: int,
        end: int,
        close_bracket: str,
        step: int,
        batch_size: int,
    ) -> list[int]:
        resolved_start = cls._resolve_index(start, batch_size)
        resolved_end = cls._resolve_index(end, batch_size)
        direction = 1 if resolved_start <= resolved_end else -1
        actual_step = max(1, step)

        first = resolved_start if open_bracket == "[" else resolved_start + direction
        last = resolved_end if close_bracket == "]" else resolved_end - direction

        return cls._build_range(first, last, actual_step, direction)

    @classmethod
    def _expand_dash_range(
        cls, start: int, end: int, step: int, batch_size: int
    ) -> list[int]:
        resolved_start = cls._resolve_index(start, batch_size)
        resolved_end = cls._resolve_index(end, batch_size)
        direction = 1 if resolved_start <= resolved_end else -1
        actual_step = max(1, step)
        return cls._build_range(resolved_start, resolved_end, actual_step, direction)

    @staticmethod
    def _build_range(first: int, last: int, step: int, direction: int) -> list[int]:
        if direction > 0:
            if first > last:
                return []
            return list(range(first, last + 1, step))

        if first < last:
            return []
        return list(range(first, last - 1, -step))
