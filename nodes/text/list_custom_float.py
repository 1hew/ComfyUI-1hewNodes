import re

from comfy_api.latest import io

from ...utils import make_ui_text


class ListCustomFloat(io.ComfyNode):
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
        items = cls._split_text(text)
        if not items:
            items = [0.0]
        count = len(items)
        return io.NodeOutput(items, count, ui=make_ui_text(str(count)))

    @staticmethod
    def _parse_section(text: str) -> list[float]:
        if not text.strip():
            return []
        lines = text.split("\n")
        float_list: list[float] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "," in line or ";" in line or "，" in line or "；" in line:
                line = (
                    line.replace(";", ",")
                    .replace("，", ",")
                    .replace("；", ",")
                )
                items = line.split(",")
                for item in items:
                    item = item.strip()
                    if (item.startswith('"') and item.endswith('"')) or (
                        item.startswith("'") and item.endswith("'")
                    ):
                        item = item[1:-1]
                    if item:
                        try:
                            float_list.append(float(item))
                        except (ValueError, TypeError):
                            continue
            else:
                if (line.startswith('"') and line.endswith('"')) or (
                    line.startswith("'") and line.endswith("'")
                ):
                    line = line[1:-1]
                if line:
                    try:
                        float_list.append(float(line))
                    except (ValueError, TypeError):
                        continue
        return float_list if float_list else [0.0]

    @staticmethod
    def _strip_item(item: str) -> str:
        s = item.strip()
        if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ('"', "'")):
            s = s[1:-1].strip()
        return s

    @classmethod
    def _split_text(cls, text: str) -> list[float]:
        t = text or ""
        if not t.strip():
            return []

        dash_line_pattern = r"^\s*-+\s*$"
        if re.search(dash_line_pattern, t, flags=re.MULTILINE):
            parts = re.split(dash_line_pattern, t, flags=re.MULTILINE)
        elif ("\n" in t) or ("\r" in t):
            parts = re.split(r"\r?\n", t)
        elif re.search(r"(。|(?<!\d)\.(?!\d))", t):
            parts = re.split(r"(?:。|(?<!\d)\.(?!\d))+", t)
        elif ("；" in t) or (";" in t):
            parts = re.split(r"[；;]+", t)
        else:
            parts = re.split(r"[，,]+", t)

        out: list[float] = []
        for p in parts:
            s = cls._strip_item(p)
            if s:
                try:
                    out.append(float(s))
                except (ValueError, TypeError):
                    continue
        return out