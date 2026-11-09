"""Load one system prompt from ``utils/system_prompt/`` by name.

The dropdown mirrors the folder tree: every ``.txt` file under
``utils/system_prompt/`` is offered as ``<model>/[<lang>] <name>``, and the
selected entry's body is returned verbatim on the ``string`` output.
"""

from comfy_api.latest import io

from ...utils._system_prompt import list_system_prompts, load_system_prompt

_EMPTY_OPTION = "(no system prompts found)"


class TextSystemPrompt(io.ComfyNode):
    """Return one builtin system prompt as a plain string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = list_system_prompts() or [_EMPTY_OPTION]
        return io.Schema(
            node_id="1hew_TextSystemPrompt",
            display_name="Text System Prompt",
            category="1hewNodes/text",
            description=(
                "Pick a builtin system prompt from utils/system_prompt and output "
                "its text. The dropdown mirrors the folder tree "
                "(model/[lang] name); the body is returned verbatim."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=options,
                    default=options[0],
                    tooltip=(
                        "Path under utils/system_prompt without the .txt suffix, "
                        "e.g. qwen_image_2.1/[en] t2i"
                    ),
                ),
            ],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    async def execute(cls, file: str) -> io.NodeOutput:
        return io.NodeOutput(load_system_prompt(file))
