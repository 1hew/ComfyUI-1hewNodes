"""Audio regression tests; also runnable directly in ComfyUI's Python."""
import asyncio
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
try:
    import torchaudio
    from torchcodec.encoders import AudioEncoder  # noqa: F401
except (ImportError, RuntimeError, OSError):
    torchaudio = None

import av
import torch

import _node_loader


def load_node(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory
    cli = types.ModuleType("comfy.cli_args")
    cli.args = types.SimpleNamespace(disable_metadata=True)
    latest = types.ModuleType("comfy_api.latest")
    latest.io = types.SimpleNamespace(
        ComfyNode=object, FolderType=types.SimpleNamespace(output="output", temp="temp"),
        NodeOutput=lambda *values, **kwargs: values,
    )
    latest.ui = types.SimpleNamespace(
        PreviewVideo=lambda values: values, SavedResult=lambda *values: values,
    )
    with patch.dict(sys.modules, {
        "folder_paths": folders, "comfy": types.ModuleType("comfy"),
        "comfy.cli_args": cli, "comfy_api.latest": latest,
    }):
        module = _node_loader.load_node(
            "audio_test_node", "nodes/io/save_video_by_image.py"
        )
    return module


@unittest.skipIf(torchaudio is None, "Requires working TorchAudio/TorchCodec; run in ComfyUI Python")
class SaveVideoAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.module = load_node(self.directory)

    def audio(self, channels=1, rate=44100):
        signal = torch.sin(torch.arange(rate, dtype=torch.float32) * (440 * 2 * 3.14159265 / rate)) * 0.5
        return {"waveform": signal.repeat(1, channels, 1), "sample_rate": rate}

    def save(self, audio=None, alpha=False, save_output=True):
        return asyncio.run(self.module.SaveVideoByImage.execute(
            image=torch.ones((8, 32, 32, 4 if alpha else 3)), fps=8.0,
            filename="test", auto_increment=False, save_output=save_output,
            save_metadata=False, audio=audio,
        ))[0]

    def assert_audible(self, path, codec, rate=None):
        with av.open(str(path)) as container:
            self.assertEqual(len(container.streams.video), 1)
            self.assertEqual(len(container.streams.audio), 1)
            stream = container.streams.audio[0]
            self.assertEqual(stream.codec_context.name, codec)
            if rate:
                self.assertEqual(stream.rate, rate)
            frames = list(container.decode(audio=0))
            self.assertTrue(frames)
            self.assertGreater(max(float(abs(f.to_ndarray()).max()) for f in frames), 0.1)
        self.assertFalse(list(Path(self.directory).glob("*_audio.wav")))

    def test_mp4_mono_and_stereo(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                self.assert_audible(self.save(self.audio(channels)), "aac", 44100)

    def test_webm_with_supported_opus_rate(self):
        self.assert_audible(self.save(self.audio(2, rate=48000), alpha=True, save_output=False), "opus", 48000)

    def test_alpha_mov_and_webm_preview_have_audio(self):
        self.assert_audible(self.save(self.audio(2, rate=48000), alpha=True), "aac", 48000)
        self.assert_audible(Path(self.directory) / "test.webm", "opus", 48000)

    def test_no_audio_remains_supported(self):
        with av.open(self.save()) as container:
            self.assertEqual(len(container.streams.audio), 0)
            self.assertEqual(len(container.streams.video), 1)

    def test_invalid_audio_raises_and_cleans_temp_wav(self):
        with self.assertRaisesRegex(RuntimeError, "Failed to prepare video audio"):
            self.save({})
        self.assertFalse(list(Path(self.directory).glob("*_audio.wav")))
        self.assertFalse((Path(self.directory) / "test.mp4").exists())

    def test_dependency_failure_is_reported(self):
        with patch.object(self.module.torchaudio, "save", side_effect=ImportError("TorchCodec unavailable")):
            with self.assertRaisesRegex(RuntimeError, "TorchCodec unavailable"):
                self.save(self.audio())
        self.assertFalse(list(Path(self.directory).glob("*_audio.wav")))

    def test_original_torchaudio_wav_save(self):
        audio = self.audio(2)
        path = str(Path(self.directory) / "check.wav")
        torchaudio.save(path, audio["waveform"].squeeze(0), audio["sample_rate"])
        with av.open(path) as container:
            stream = container.streams.audio[0]
            self.assertEqual(stream.rate, 44100)
            self.assertEqual(len(stream.layout.channels), 2)
            self.assertGreater(float(abs(next(container.decode(audio=0)).to_ndarray()).max()), 0.1)

    def test_missing_audio_file_does_not_silently_succeed(self):
        path = str(Path(self.directory) / "broken.mp4")
        with self.assertRaises(Exception):
            asyncio.run(self.module.SaveVideoByImage.ffmpeg_encode(
                images=torch.ones(8, 32, 32, 3), path=path, fps=8,
                audio_path=str(Path(self.directory) / "missing.wav"), width=32, height=32,
                input_pix_fmt="rgb24", video_codec="libx264", output_pix_fmt="yuv420p",
                video_args=[], audio_codec="aac",
            ))
        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
