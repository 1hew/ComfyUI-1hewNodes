import re

from comfy_api.latest import io

from ...utils import make_ui_text


class ListCustomFloat(io.ComfyNode):
    _NUMBER_PATTERN = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
    _BRACKET_RANGE_RE = re.compile(
        rf"(?P<open>[\[\(])\s*(?P<start>{_NUMBER_PATTERN})\s*,\s*(?P<end>{_NUMBER_PATTERN})\s*(?P<close>[\]\)])\s*(?:\:\s*(?P<step>{_NUMBER_PATTERN}))?"
    )
    _DASH_RANGE_RE = re.compile(
        rf"(?P<start>{_NUMBER_PATTERN})\s*-\s*(?P<end>{_NUMBER_PATTERN})\s*(?:\:\s*(?P<step>{_NUMBER_PATTERN}))?"
    )
    _NUMBER_RE = re.compile(rf"(?P<value>{_NUMBER_PATTERN})")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListCustomFloat",
            display_name="List Custom Float",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("custom_text", default="", multiline=True),
            ],
            outputs=[
                io.Float.Output(display_name="float_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        items = cls._parse_values(text)
        if not items:
            items = [0.0]
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
    def _parse_values(cls, text: str) -> list[float]:
        normalized = cls._normalize_text(text)
        if not normalized.strip():
            return []

        out: list[float] = []
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
                        float(match.group("start")),
                        float(match.group("end")),
                        match.group("close"),
                        float(match.group("step")) if match.group("step") else 1.0,
                    )
                )
                pos = match.end()
                continue

            match = cls._DASH_RANGE_RE.match(normalized, pos)
            if match:
                out.extend(
                    cls._expand_dash_range(
                        float(match.group("start")),
                        float(match.group("end")),
                        float(match.group("step")) if match.group("step") else 1.0,
                    )
                )
                pos = match.end()
                continue

            match = cls._NUMBER_RE.match(normalized, pos)
            if match:
                try:
                    out.append(cls._clean_float(float(match.group("value"))))
                except (ValueError, TypeError):
                    pass
                pos = match.end()
                continue

            pos += 1

        return out

    @classmethod
    def _expand_bracket_range(
        cls,
        open_bracket: str,
        start: float,
        end: float,
        close_bracket: str,
        step: float,
    ) -> list[float]:
        direction = 1 if start <= end else -1
        actual_step = abs(step)
        if actual_step <= 0:
            return []

        first = start if open_bracket == "[" else start + (direction * actual_step)
        last = end if close_bracket == "]" else end - (direction * actual_step)

        return cls._build_range(first, last, actual_step, direction)

    @classmethod
    def _expand_dash_range(cls, start: float, end: float, step: float) -> list[float]:
        actual_step = abs(step)
        if actual_step <= 0:
            return []
        direction = 1 if start <= end else -1
        return cls._build_range(start, end, actual_step, direction)

    @classmethod
    def _build_range(
        cls, first: float, last: float, step: float, direction: int
    ) -> list[float]:
        values: list[float] = []
        tolerance = max(1e-12, step * 1e-9)
        current = first

        if direction > 0:
            while current <= last + tolerance:
                values.append(cls._clean_float(current))
                current += step
            return values

        while current >= last - tolerance:
            values.append(cls._clean_float(current))
            current -= step
        return values

    @staticmethod
    def _clean_float(value: float) -> float:
        return float(f"{value:.12f}")