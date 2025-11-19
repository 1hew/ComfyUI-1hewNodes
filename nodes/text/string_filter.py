from comfy_api.latest import io


class StringFilter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringFilter",
            display_name="String Filter",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.Boolean.Input("filter_empty_line", default=False),
                io.Boolean.Input("filter_comment", default=False),
                io.String.Input("input", default=""),
            ],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    async def execute(
        cls,
        text: str,
        filter_empty_line: bool,
        filter_comment: bool,
        input: str,
        ) -> io.NodeOutput:
        text = (text or "").replace("\\n", "\n")
        in_val = "" if input is None else str(input)
        text = text.replace("{input}", in_val)
        filtered_text = cls._parse_text_and_filter_comments(
            text, filter_comment, filter_empty_line
        )
        return io.NodeOutput(str(filtered_text))

    @classmethod
    def _parse_text_and_filter_comments(
        cls, text: str, filter_comment: bool, filter_empty_line: bool
    ) -> str:
        if not text or not text.strip():
            return ""

        lines = text.split("\n")
        result = []
        in_multiline_comment = False
        multiline_quote_type = None

        for line in lines:
            original_line = line
            processed_line = ""
            if filter_comment:
                i = 0
                while i < len(line):
                    if in_multiline_comment:
                        end_pos = line.find(multiline_quote_type, i)
                        if end_pos != -1:
                            i = end_pos + len(multiline_quote_type)
                            in_multiline_comment = False
                            multiline_quote_type = None
                        else:
                            break
                    else:
                        if (
                            i + 2 < len(line)
                            and (
                                line[i : i + 3] == '"""'
                                or line[i : i + 3] == "'''"
                            )
                        ):
                            multiline_quote_type = line[i : i + 3]
                            end_pos = line.find(multiline_quote_type, i + 3)
                            if end_pos != -1:
                                i = end_pos + 3
                            else:
                                in_multiline_comment = True
                                break
                        elif line[i] == '#':
                            break
                        else:
                            processed_line += line[i]
                            i += 1
            else:
                processed_line = original_line

            if not in_multiline_comment:
                processed_line = processed_line.rstrip()

                if filter_comment:
                    starts_with_comment = original_line.lstrip().startswith('#')
                    became_empty_by_filter = (
                        processed_line.strip() == ""
                        and original_line.strip() != ""
                        and (
                            '#' in original_line
                            or original_line.strip().startswith('"""')
                            or original_line.strip().startswith("'''")
                        )
                    )
                    if starts_with_comment or became_empty_by_filter:
                        continue

                if filter_empty_line:
                    if processed_line.strip():
                        result.append(processed_line)
                else:
                    result.append(processed_line)

        parsed_text = "\n".join(result)
        if not parsed_text.strip():
            return ""
        return parsed_text

