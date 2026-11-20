"""List Text: apply a prefix and suffix to every item of a list, keeping it a list.

Same prefix / suffix inputs as Text List to String. The difference is the
output: Text List to String joins the items into one string, this node keeps
one string per item so the result can feed the next list node.
"""

from comfy_api.latest import io

from ...utils import make_ui_text


class ListText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListText",
            display_name="List Text",
            category="1hewNodes/list",
            is_input_list=True,
            inputs=[
                io.Custom("*").Input("text", optional=True),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
            ],
            outputs=[
                io.String.Output(display_name="text", is_output_list=True),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls, prefix: str, suffix: str, text=None,
    ) -> io.NodeOutput:
        # is_input_list wraps widget values in a list; unwrap the single item.
        if isinstance(prefix, (list, tuple)):
            prefix = prefix[0] if len(prefix) > 0 else ""
        if isinstance(suffix, (list, tuple)):
            suffix = suffix[0] if len(suffix) > 0 else ""
        prefix = "" if prefix is None else str(prefix)
        suffix = "" if suffix is None else str(suffix)

        # Same flattening as Text List to String: one level of nested lists.
        if text is None:
            items = []
        elif isinstance(text, (list, tuple)):
            items = []
            for item in text:
                if isinstance(item, (list, tuple)):
                    items.extend(item)
                else:
                    items.append(item)
        else:
            items = [text]

        result = [f"{prefix}{str(item)}{suffix}" for item in items]
        return io.NodeOutput(result, ui=make_ui_text(str(len(result))))
