from comfy_api.latest import io

class MultiStringJoin(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_MultiStringJoin",
            display_name="Multi String Join",
            category="1hewNodes/multi",
            inputs=[
                io.Boolean.Input("filter_empty_line", default=False),
                io.Boolean.Input("filter_comment", default=False),
                io.String.Input("separator", default="\\n"),
                io.String.Input("input", default=""),
            ],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    def _parse_text_with_input(
        cls,
        text,
        input,
        filter_comment=False,
        filter_empty_line=False,
    ):
        safe_input = "" if input is None else str(input)
        parsed = "" if text is None else str(text)
        parsed = parsed.replace("{input}", safe_input)

        lines = parsed.split("\n")
        result = []
        in_block = False
        block_quote = None
        for line in lines:
            original = line
            processed = ""

            if filter_comment:
                i = 0
                while i < len(line):
                    if in_block:
                        end_pos = line.find(block_quote, i)
                        if end_pos != -1:
                            i = end_pos + len(block_quote)
                            in_block = False
                            block_quote = None
                        else:
                            break
                    else:
                        is_triple = (
                            i + 2 < len(line)
                            and (
                                line[i : i + 3] == '"""'
                                or line[i : i + 3] == "'''"
                            )
                        )
                        if is_triple:
                            block_quote = line[i : i + 3]
                            end_pos = line.find(block_quote, i + 3)
                            if end_pos != -1:
                                i = end_pos + 3
                            else:
                                in_block = True
                                break
                        elif line[i] == '#':
                            break
                        else:
                            processed += line[i]
                            i += 1
            else:
                processed = original

            if not in_block:
                processed = processed.rstrip()
                if filter_comment:
                    starts_with_comment = original.lstrip().startswith('#')
                    became_empty_by_filter = (
                        processed.strip() == ""
                        and original.strip() != ""
                        and (
                            '#' in original
                            or original.strip().startswith('"""')
                            or original.strip().startswith("'''")
                        )
                    )
                    if starts_with_comment or became_empty_by_filter:
                        continue

                if filter_empty_line:
                    if processed.strip():
                        result.append(processed)
                else:
                    result.append(processed)

        final_text = "\n".join(result)
        return "" if not final_text.strip() else final_text

    @classmethod
    async def execute(
        cls,
        filter_empty_line,
        filter_comment,
        separator,
        input,
        **kwargs,
    ) -> io.NodeOutput:
        sep = "" if separator is None else str(separator)
        sep = (
            sep.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
        )

        ordered = []
        for k in kwargs.keys():
            if k.startswith("string_"):
                suf = k[len("string_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        parts = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            parsed = cls._parse_text_with_input(
                val,
                input,
                filter_comment,
                filter_empty_line,
            )
            if parsed.strip():
                parts.append(parsed)

        result = sep.join(parts)
        safe_input = "" if input is None else str(input)
        result = result.replace("{input}", safe_input)
        return io.NodeOutput(str(result))
