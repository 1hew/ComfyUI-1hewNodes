from comfy_api.latest import io


class TextMatchValue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_text_match_value",
            display_name="Text Match Value",
            category="1hewNodes/logic",
            inputs=[
                io.String.Input("text_multiline", multiline=True),
                io.String.Input("text_single", multiline=False),
            ],
            outputs=[io.Custom("*").Output(display_name="value")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, text_multiline: str, text_single: str) -> io.NodeOutput:
        try:
            query = cls._normalize(text_single)
            if not query:
                return io.NodeOutput("")

            entries = cls._parse_entries(text_multiline)

            for key, value in entries:
                if cls._normalize(key) == query:
                    return io.NodeOutput(value)

            for key, value in entries:
                normalized_key = cls._normalize(key)
                if normalized_key.startswith(query):
                    return io.NodeOutput(value)

            return io.NodeOutput("")
        except Exception as e:
            print(f"text_match_value error: {e}")
            return io.NodeOutput("")

    @staticmethod
    def _normalize(text: str) -> str:
        return str(text or "").strip().lower()

    @classmethod
    def _parse_entries(cls, text_multiline: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for raw_line in (text_multiline or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = cls._split_mapping_line(line)
            if parts is None:
                continue
            key, value = parts

            key = cls._unwrap_braces(key.strip())
            value = cls._unwrap_braces(value.strip())
            if not key:
                continue

            entries.append((key, value))
        return entries

    @staticmethod
    def _split_mapping_line(line: str) -> tuple[str, str] | None:
        depth = 0
        for idx, char in enumerate(line):
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth = max(0, depth - 1)
                continue
            if depth == 0 and char in (":", "："):
                return line[:idx], line[idx + 1 :]
        return None

    @classmethod
    def _unwrap_braces(cls, text: str) -> str:
        if len(text) >= 2 and text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            if cls._is_balanced_braces(inner):
                return inner
        return text

    @staticmethod
    def _is_balanced_braces(text: str) -> bool:
        depth = 0
        for char in text:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0
