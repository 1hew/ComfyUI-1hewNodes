from comfy_api.latest import io

from ...utils._index_range import normalize_text, parse_index_expressions


class MultiSwitchSelect(io.ComfyNode):
    SELECT_MAX = 10
    # Whole-token shortcuts for "every connected input". Handled here instead
    # of in the shared parser so Image Batch Index keeps its numeric grammar.
    SELECT_ALL_KEYWORDS = frozenset({"a", "all"})

    @classmethod
    def define_schema(cls):
        # Every dynamic port must be declared here, even though the front-end
        # adds input_2..input_10 on demand. ComfyUI's TopologicalSort only
        # skips an upstream branch when the port's schema entry carries
        # "lazy"; undeclared inputs have no schema info and are treated as
        # eager, which schedules every connected branch at once.
        inputs = [
            io.String.Input(
                "select",
                default="1",
                tooltip=(
                    "1-based indices, same syntax as Image Batch Index: "
                    "1,3 / 1-3 / [1,3] / [1,3) / :step; "
                    "[1:3] means 1 and 3; -1 = last connected input; "
                    "a / all = every connected input."
                ),
            ),
            io.Custom("*").Input("input_1", optional=True, lazy=True),
        ]
        for i in range(2, cls.SELECT_MAX + 1):
            inputs.append(
                io.Custom("*").Input(f"input_{i}", optional=True, lazy=True)
            )
        return io.Schema(
            node_id="1hew_MultiSwitchSelect",
            display_name="Multi Switch Select",
            category="1hewNodes/logic",
            inputs=inputs,
            outputs=[
                io.Custom("*").Output(display_name=f"output_{i}")
                for i in range(1, cls.SELECT_MAX + 1)
            ],
        )

    @classmethod
    def validate_inputs(cls, input_types=None, **kwargs):
        # Declaring an explicit `input_types` argument tells ComfyUI to skip the
        # linked-type check for this node. That keeps legacy workflows and API
        # prompts working when `select` is driven by an INT node; execute()
        # accepts both integers and index-list strings.
        return True

    @classmethod
    def check_lazy_status(cls, select, **kwargs):
        size = cls._connected_size(kwargs)
        # Request every selected input that is connected but not resolved yet.
        # Unconnected ports must be skipped: requesting one raises NodeInputError.
        return [
            key
            for key in cls._selected_keys(select, size)
            if key in kwargs and kwargs.get(key) is None
        ]

    @classmethod
    async def execute(cls, select, **kwargs) -> io.NodeOutput:
        results = [None] * cls.SELECT_MAX
        for index in cls._selected_indices(select, cls._connected_size(kwargs)):
            results[index - 1] = kwargs.get(f"input_{index}")
        return io.NodeOutput(*results)

    @classmethod
    def _selected_indices(cls, select, size=None):
        size = cls.SELECT_MAX if size is None else max(0, int(size))

        def resolve(value):
            # Image Batch Index resolves negatives against the batch size; the
            # analogous 1-based base here is the highest connected input slot.
            if value < 0:
                return size + value + 1
            return value

        if isinstance(select, bool):
            select = int(select)
        if isinstance(select, (int, float)):
            values = [resolve(int(select))]
        else:
            if isinstance(select, (list, tuple, set)):
                raw = ",".join(str(item) for item in select)
            else:
                raw = "" if select is None else str(select)
            if cls._selects_all(raw):
                return list(range(1, size + 1))
            values = parse_index_expressions(raw, resolve)

        return sorted({value for value in values if 1 <= value <= size})

    @classmethod
    def _selects_all(cls, raw):
        # Only a standalone token counts: "a1" still means index 1, not all.
        text = normalize_text(str(raw)).lower().replace(";", " ").replace(",", " ")
        return any(token in cls.SELECT_ALL_KEYWORDS for token in text.split())

    @classmethod
    def _selected_keys(cls, select, size=None):
        return [f"input_{index}" for index in cls._selected_indices(select, size)]

    @staticmethod
    def _connected_size(kwargs):
        size = 0
        for key in kwargs:
            if key.startswith("input_"):
                suffix = key[len("input_") :]
                if suffix.isdigit():
                    size = max(size, int(suffix))
        return size
