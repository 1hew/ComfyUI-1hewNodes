import re

from comfy_api.latest import io


class ListCustomString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListCustomString",
            display_name="List Custom String",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("custom_text", default="", multiline=True),
            ],
            outputs=[
                io.MultiType.Output(display_name="string_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        if not text.strip():
            string_list = ["default"]
        else:
            lines = text.split("\n")
            has_dash = any(
                line.strip() and all(c == "-" for c in line.strip())
                for line in lines
            )
            if has_dash:
                sections = re.split(r"^\s*-+\s*$", text, flags=re.MULTILINE)
                all_lists: list[str] = []
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    if (
                        (section.startswith('"') and section.endswith('"'))
                        or (section.startswith("'") and section.endswith("'"))
                    ):
                        section = section[1:-1]
                    if section:
                        all_lists.append(str(section))
                string_list = all_lists if all_lists else ["default"]
            else:
                string_list = cls._parse_section(text)
        return io.NodeOutput(string_list, len(string_list))

    @staticmethod
    def _parse_section(text: str) -> list[str]:
        if not text.strip():
            return []
        lines = text.split("\n")
        items_out: list[str] = []
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
                        items_out.append(str(item))
            else:
                if (line.startswith('"') and line.endswith('"')) or (
                    line.startswith("'") and line.endswith("'")
                ):
                    line = line[1:-1]
                if line:
                    items_out.append(str(line))
        return items_out if items_out else ["default"]
