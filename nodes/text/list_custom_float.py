import re

from comfy_api.latest import io


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
                io.MultiType.Output(display_name="float_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        if not text.strip():
            float_list = [0.0]
        else:
            lines = text.split("\n")
            has_dash = any(
                line.strip() and all(c == "-" for c in line.strip())
                for line in lines
            )
            if has_dash:
                sections = re.split(r"^\s*-+\s*$", text, flags=re.MULTILINE)
                all_lists: list[float] = []
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    if (
                        (section.startswith('"') and section.endswith('"'))
                        or (section.startswith("'") and section.endswith("'"))
                    ):
                        section = section[1:-1]
                    try:
                        all_lists.append(float(section))
                    except (ValueError, TypeError):
                        continue
                float_list = all_lists if all_lists else [0.0]
            else:
                float_list = cls._parse_section(text)
        return io.NodeOutput(float_list, len(float_list))

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