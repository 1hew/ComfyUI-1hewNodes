import re

from comfy_api.latest import io

from ...utils import make_ui_text


class ListCustomInt(io.ComfyNode):
    _BRACKET_RANGE_RE = re.compile(
        r"(?P<open>[\[\(])\s*(?P<start>-?\d+)\s*,\s*(?P<end>-?\d+)\s*(?P<close>[\]\)])\s*(?:\:\s*(?P<step>\d+))?"
    )
    _DASH_RANGE_RE = re.compile(
        r"(?P<start>-?\d+)\s*-\s*(?P<end>-?\d+)\s*(?:\:\s*(?P<step>\d+))?"
    )
    _NUMBER_RE = re.compile(r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+))")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListCustomInt",
            display_name="List Custom Int",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("custom_text", default="", multiline=True),
            ],
            outputs=[
                io.Int.Output(display_name="int_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        items = cls._parse_values(text)
        if not items:
            items = [0]
        count = len(items)
        return io.NodeOutput(items, count, ui=make_ui_text(str(count)))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (
            (text or "")
            .replace("【", "[")
            .replace("】", "]")
            .replace("（", "(")
            .replace("）", ")")
            .replace("，", ",")
            .replace("：", ":")
            .replace("；", ",")
            .replace(";", ",")
        )

    @classmethod
    def _parse_values(cls, text: str) -> list[int]:
        normalized = cls._normalize_text(text)
        if not normalized.strip():
            return []

        out: list[int] = []
        pos = 0
        length = len(normalized)

        while pos < length:
            ch = normalized[pos]
            if ch.isspace() or ch == ",":
                pos += 1
                continue

            match = cls._BRACKET_RANGE_RE.match(normalized, pos)
            if match:
                out.extend(
                    cls._expand_bracket_range(
                        match.group("open"),
                        int(match.group("start")),
                        int(match.group("end")),
                        match.group("close"),
                        int(match.group("step")) if match.group("step") else 1,
                    )
                )
                pos = match.end()
                continue

            match = cls._DASH_RANGE_RE.match(normalized, pos)
            if match:
                out.extend(
                    cls._expand_dash_range(
                        int(match.group("start")),
                        int(match.group("end")),
                        int(match.group("step")) if match.group("step") else 1,
                    )
                )
                pos = match.end()
                continue

            match = cls._NUMBER_RE.match(normalized, pos)
            if match:
                try:
                    out.append(int(float(match.group("value"))))
                except (ValueError, TypeError):
                    pass
                pos = match.end()
                continue

            pos += 1

        return out

    @classmethod
    def _expand_bracket_range(
        cls, open_bracket: str, start: int, end: int, close_bracket: str, step: int
    ) -> list[int]:
        direction = 1 if start <= end else -1
        actual_step = max(1, step)

        first = start if open_bracket == "[" else start + direction
        last = end if close_bracket == "]" else end - direction

        return cls._build_range(first, last, actual_step, direction)

    @classmethod
    def _expand_dash_range(cls, start: int, end: int, step: int) -> list[int]:
        direction = 1 if start <= end else -1
        return cls._build_range(start, end, max(1, step), direction)

    @staticmethod
    def _build_range(first: int, last: int, step: int, direction: int) -> list[int]:
        if direction > 0:
            if first > last:
                return []
            return list(range(first, last + 1, step))

        if first < last:
            return []
        return list(range(first, last - 1, -step))
