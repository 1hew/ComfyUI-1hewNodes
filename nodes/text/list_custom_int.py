import re

from comfy_api.latest import io


class ListCustomInt(io.ComfyNode):
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
                io.MultiType.Output(display_name="int_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        if not text.strip():
            int_list = [0]
        else:
            lines = text.split("\n")
            has_dash = any(
                line.strip() and all(c == "-" for c in line.strip())
                for line in lines
            )
            if has_dash:
                sections = re.split(r"^\s*-+\s*$", text, flags=re.MULTILINE)
                all_lists: list[int] = []
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
                        if "." in section:
                            int_val = int(float(section))
                        else:
                            int_val = int(section)
                        all_lists.append(int_val)
                    except (ValueError, TypeError):
                        continue
                int_list = all_lists if all_lists else [0]
            else:
                int_list = cls._parse_section(text)
        return io.NodeOutput(int_list, len(int_list))

    @staticmethod
    def _parse_section(text: str) -> list[int]:
        if not text.strip():
            return []
        lines = text.split("\n")
        int_list: list[int] = []
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
                            if "." in item:
                                int_val = int(float(item))
                            else:
                                int_val = int(item)
                            int_list.append(int_val)
                        except (ValueError, TypeError):
                            continue
            else:
                if (line.startswith('"') and line.endswith('"')) or (
                    line.startswith("'") and line.endswith("'")
                ):
                    line = line[1:-1]
                if line:
                    try:
                        if "." in line:
                            int_val = int(float(line))
                        else:
                            int_val = int(line)
                        int_list.append(int_val)
                    except (ValueError, TypeError):
                        continue
        return int_list if int_list else [0]
