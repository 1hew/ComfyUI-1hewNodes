from comfy_api.latest import io


class AudioDuration(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AudioDuration",
            display_name="Audio Duration",
            category="1hewNodes/audio",
            inputs=[io.Audio.Input("audio")],
            outputs=[io.Float.Output(display_name="second")],
        )

    @classmethod
    async def execute(cls, audio: dict) -> io.NodeOutput:
        if not audio or not isinstance(audio, dict):
            return io.NodeOutput(0.0)
        sr = float(audio.get("sample_rate") or 0)
        wf = audio.get("waveform")
        if sr <= 0 or wf is None:
            return io.NodeOutput(0.0)
        try:
            shp = getattr(wf, "shape", None)
            if not shp or len(shp) < 3:
                return io.NodeOutput(0.0)
            samples = int(shp[2])
            secs = float(samples) / float(sr)
            return io.NodeOutput(secs)
        except Exception:
            return io.NodeOutput(0.0)