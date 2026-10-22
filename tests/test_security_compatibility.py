"""Offline CPU checks for the legacy dependency refresh, without model weights."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from click.testing import CliRunner
from moviepy import VideoFileClip
from genmo.lib.utils import save_video


class DependencyCompatibility(unittest.TestCase):
    def test_video_export_trim_crop_and_caption(self):
        spec = importlib.util.spec_from_file_location('trim', ROOT / 'demos/fine_tuner/trim_and_crop_videos.py')
        trim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trim)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            target = Path(directory) / 'target'
            source.mkdir()
            # Exercise both width and height crop branches with Pillow 13.
            for name, width, height in [('wide', 96, 48), ('tall', 48, 96)]:
                frames = np.zeros((12, height, width, 3), dtype=np.float32)
                frames[..., 0] = 0.75
                save_video(frames, str(source / f'{name}.mp4'), fps=24)
                (source / f'{name}.txt').write_text(name)
            result = CliRunner().invoke(trim.truncate_videos, [str(source), str(target), '--duration', '0.2', '--resolution', '32x32'])
            self.assertEqual(result.exit_code, 0, result.output)
            for name in ['wide', 'tall']:
                with VideoFileClip(str(target / f'{name}.mp4')) as video:
                    self.assertEqual(video.size, [32, 32])
                    self.assertEqual(video.fps, 30)
                    self.assertAlmostEqual(video.duration, 0.2, places=1)
                    self.assertGreater(video.get_frame(0)[16, 16, 0], 150)
                self.assertEqual((target / f'{name}.txt').read_text(), name)

    def test_gradio_interface_constructs_without_loading_weights(self):
        cli = types.ModuleType('cli')
        cli.configure_model = lambda *args: None
        cli.generate_video = lambda prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_inference_steps: None
        with patch.dict(sys.modules, {'cli': cli}):
            ui = runpy.run_path(str(ROOT / 'demos/gradio_ui.py'))
        self.assertEqual(len(ui['demo'].config['dependencies']), 1)
        ui['demo'].close()

    def test_t5_encoder_runs_with_transformers_without_remote_assets(self):
        import torch
        from transformers import T5Config, T5EncoderModel
        from transformers.models.t5.modeling_t5 import T5Block
        model = T5EncoderModel(T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8))
        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 3, 16))
        self.assertIsInstance(model.encoder.block[0], T5Block)


if __name__ == '__main__':
    unittest.main()
