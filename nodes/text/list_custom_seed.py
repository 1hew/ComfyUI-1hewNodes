import random

from comfy_api.latest import io

from ...utils import make_ui_text


class ListCustomSeed(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListCustomSeed",
            display_name="List Custom Seed",
            category="1hewNodes/text",
            inputs=[
                io.Int.Input("seed", default=42, min=0, max=1125899906842624, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("count", default=3, min=1, max=1000, step=1),
            ],
            outputs=[
                io.Int.Output(display_name="seed_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, seed: int, count: int) -> io.NodeOutput:
        random.seed(int(seed))
        total = int(count)
        seeds: list[int] = []
        used: set[int] = set()
        max_attempts = max(total * 10, 100)
        attempts = 0
        while len(seeds) < total and attempts < max_attempts:
            s = random.randint(0, 1125899906842624)
            if s not in used:
                used.add(s)
                seeds.append(cls._clamp_seed(s))
            attempts += 1
        if len(seeds) < total:
            while len(seeds) < total:
                s = random.randint(0, 1125899906842624)
                if s not in used:
                    used.add(s)
                    seeds.append(cls._clamp_seed(s))
        count_val = len(seeds)
        return io.NodeOutput(seeds, count_val, ui=make_ui_text(str(count_val)))

    @staticmethod
    def _clamp_seed(seed: int) -> int:
        return max(0, min(seed, 1125899906842624))