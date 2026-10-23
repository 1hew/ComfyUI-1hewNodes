"""Save Audio tests: optional/empty input handling and real encoding.

Runs standalone (no ComfyUI needed) by stubbing folder_paths and comfy_api.
Regression target: an unconnected optional AUDIO port is not serialized into
the prompt, so execute() must tolerate a missing keyword and a None audio.
"""

import asyncio
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import torch

import _node_loader

ROOT = Path(__file__).parents[1]


def _port(name):
    class _Port:
        def Input(self, *args, **kwargs):
            return ("input", name, args, kwargs)

        def Output(self, *args, **kwargs):
            return ("output", name, args, kwargs)

    return _Port()


class _NodeOutput:
    def __init__(self, *values, ui=None, **kwargs):
        self.values = values
        self.ui = ui


def load_node(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory
    folders.get_save_image_path = lambda prefix, base: (
        base,
        "ComfyUI",
        1,
        "",
        prefix,
    )

    cli = types.ModuleType("comfy.cli_args")
    cli.args = types.SimpleNamespace(disable_metadata=True)

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=_NodeOutput,
        FolderType=types.SimpleNamespace(output="output", temp="temp"),
        Hidden=types.SimpleNamespace(prompt="PROMPT", extra_pnginfo="EXTRA_PNGINFO"),
        Audio=_port("AUDIO"),
        String=_port("STRING"),
        Combo=_port("COMBO"),
        Boolean=_port("BOOLEAN"),
    )
    ui = types.SimpleNamespace(
        SavedResult=lambda *values: values,
        SavedAudios=lambda results: ("saved-audio", results),
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui

    with patch.dict(
        sys.modules,
        {
            "folder_paths": folders,
            "comfy": types.ModuleType("comfy"),
            "comfy.cli_args": cli,
            "comfy_api": types.ModuleType("comfy_api"),
            "comfy_api.latest": latest,
        },
    ):
        module = _node_loader.load_node("save_audio_test", "nodes/io/save_audio.py")
    return module


def _audio(waveform, sample_rate=44100):
    return {"waveform": waveform, "sample_rate": sample_rate}


class SaveAudioOptionalInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.module = load_node(self.directory)

    def save(self, audio=None, **kwargs):
        params = dict(
            filename="test",
            format="wav",
            auto_increment=False,
            save_output=True,
            save_metadata=False,
        )
        params.update(kwargs)
        if audio is not None:
            params["audio"] = audio
        return asyncio.run(self.module.SaveAudio.execute(**params))

    def files(self):
        return sorted(os.listdir(self.directory))

    def test_schema_declares_optional_audio(self):
        schema = self.module.SaveAudio.define_schema()
        self.assertEqual(schema["node_id"], "1hew_SaveAudio")
        self.assertEqual(schema["display_name"], "Save Audio")
        self.assertEqual(schema["category"], "1hewNodes/io")
        audio_port = schema["inputs"][0]
        self.assertEqual(audio_port[1], "AUDIO")
        self.assertTrue(audio_port[3].get("optional"))

    def test_unknown_format_falls_back_to_mp3(self):
        result = self.save(_audio(torch.zeros(1, 1, 441)), format="bogus")
        self.assertTrue(result.values[0].endswith(".mp3"))

    def test_execute_has_python_defaults(self):
        signature = inspect.signature(self.module.SaveAudio.execute)
        self.assertIsNone(signature.parameters["audio"].default)
        self.assertEqual(signature.parameters["format"].default, "mp3")

    def test_unconnected_audio_writes_nothing(self):
        # Mirrors a prompt with the optional port left empty: no "audio" key.
        result = self.save()
        self.assertEqual(result.values, ())
        # Empty audio list clears any preview left over from a previous run.
        self.assertEqual(result.ui, ("saved-audio", []))
        self.assertEqual(self.files(), [])

    def test_bare_execute_call_does_not_raise(self):
        result = asyncio.run(self.module.SaveAudio.execute())
        self.assertEqual(result.values, ())
        self.assertEqual(self.files(), [])

    def test_invalid_or_empty_audio_is_ignored(self):
        for audio in (
            {},
            {"waveform": None, "sample_rate": 44100},
            {"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100},
            {"waveform": torch.zeros(0, 2, 16), "sample_rate": 44100},
            {"waveform": torch.zeros(1, 2, 16), "sample_rate": 0},
        ):
            with self.subTest(audio=audio):
                result = self.save(audio)
                self.assertEqual(result.values, ())
        self.assertEqual(self.files(), [])

    def test_connected_audio_saves_file(self):
        result = self.save(_audio(torch.zeros(1, 1, 4410)))
        self.assertEqual(len(result.values), 1)
        path = Path(result.values[0])
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "test.wav")
        with av.open(str(path)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(container.streams.audio[0].rate, 44100)
        self.assertEqual(result.ui[0], "saved-audio")

    def test_batch_writes_one_file_per_item(self):
        result = self.save(_audio(torch.zeros(3, 2, 2205), sample_rate=22050))
        paths = result.values[0].split("\n")
        self.assertEqual(
            sorted(Path(path).name for path in paths),
            ["test.wav", "test_001.wav", "test_002.wav"],
        )
        for path in paths:
            self.assertTrue(Path(path).is_file())

    def test_auto_increment_uses_counter(self):
        result = self.save(_audio(torch.zeros(1, 1, 441)), auto_increment=True)
        self.assertEqual(Path(result.values[0]).name, "ComfyUI_00001_.wav")

    def test_fixed_name_overwrites(self):
        first = self.save(_audio(torch.zeros(1, 1, 441)))
        second = self.save(_audio(torch.zeros(1, 1, 441)))
        self.assertEqual(first.values, second.values)
        self.assertEqual(self.files(), ["test.wav"])

    def test_absolute_prefix_is_respected(self):
        target = os.path.join(self.directory, "nested", "absolute")
        result = self.save(
            _audio(torch.zeros(1, 1, 441)), filename=target
        )
        self.assertEqual(result.values[0], os.path.abspath(target + ".wav"))
        self.assertTrue(Path(result.values[0]).is_file())

    def test_save_output_false_uses_temp_directory(self):
        result = self.save(
            _audio(torch.zeros(1, 1, 441)), save_output=False
        )
        self.assertTrue(result.values[0].startswith(os.path.abspath(self.directory)))
        self.assertEqual(result.ui[1][0][2], "temp")

    def test_multichannel_audio_fails_cleanly(self):
        with self.assertRaisesRegex(RuntimeError, "mono/stereo"):
            self.save(_audio(torch.zeros(1, 3, 441)))
        self.assertEqual(self.files(), [])

    @unittest.skipUnless("flac" in getattr(av, "codecs_available", ()), "no flac codec")
    def test_flac_roundtrip(self):
        result = self.save(_audio(torch.zeros(1, 2, 4410)), format="flac")
        self.assertTrue(result.values[0].endswith(".flac"))
        with av.open(result.values[0]) as container:
            self.assertEqual(len(container.streams.audio), 1)

    @unittest.skipUnless(
        "libopus" in getattr(av, "codecs_available", ()), "no libopus codec"
    )
    def test_opus_resamples_unsupported_rate(self):
        result = self.save(_audio(torch.zeros(1, 2, 4410), sample_rate=44100), format="opus")
        with av.open(result.values[0]) as container:
            self.assertEqual(container.streams.audio[0].rate, 48000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
