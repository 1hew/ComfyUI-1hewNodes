from comfy_api.latest import io


class TextListToString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextListToString",
            display_name="Text List to String",
            category="1hewNodes/conversion",
            is_input_list=True,
            inputs=[
                io.Custom("*").Input("text_list", optional=True),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
                io.String.Input("separator", default="\\n"),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls, prefix: str, suffix: str, separator: str, text_list=None,
    ) -> io.NodeOutput:
        if isinstance(prefix, (list, tuple)):
            prefix = prefix[0] if len(prefix) > 0 else ""
        if isinstance(suffix, (list, tuple)):
            suffix = suffix[0] if len(suffix) > 0 else ""
        prefix = "" if prefix is None else str(prefix)
        suffix = "" if suffix is None else str(suffix)

        if isinstance(separator, (list, tuple)):
            separator = separator[0] if len(separator) > 0 else "\\n"
        separator = str(separator)

        sep = (
            separator.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\\\\", "\\")
        )

        if text_list is None:
            items = []
        elif isinstance(text_list, (list, tuple)):
            items = []
            for item in text_list:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        items.append(sub)
                else:
                    items.append(item)
        else:
            items = [text_list]

        formatted = [f"{prefix}{str(item)}{suffix}" for item in items]
        joined = sep.join(formatted)
        return io.NodeOutput(str(joined))
