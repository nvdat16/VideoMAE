"""CPU smoke tests with a tiny real VideoMAE backbone, no model download."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from transformers import VideoMAEConfig, VideoMAEModel, TrainerCallback
    from videomae_local import training as tr
    from videomae_local.models import MeanPoolVideoMAEClassifier, EventLevelVideoUnderstandingModel
    HAS_ML = True
except ImportError:
    HAS_ML = False


@unittest.skipUnless(HAS_ML, 'Install requirements.txt for CPU training smoke tests')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        original_args = tr.TrainingArguments
        def cpu_args(**kwargs):
            return original_args(**kwargs, use_cpu=True, disable_tqdm=True)
        args_patch = patch.object(tr, 'TrainingArguments', side_effect=cpu_args)
        args_patch.start()
        self.addCleanup(args_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tr.TRAIN_BATCH_SIZE = tr.EVAL_BATCH_SIZE = 1
        tr.GRADIENT_ACCUMULATION_STEPS = 2
        tr.LEARNING_RATE, tr.NUM_EPOCHS, tr.WEIGHT_DECAY = 2e-5, 2, .05
        tr.NUM_WORKERS, tr.PRECISION, tr.SEED = 0, 'fp32', 42
        tr.EARLY_STOPPING_PATIENCE = 3
        tr.MODEL_OUTPUT_ROOT = self.tmp.name
        tr.train_dataset = tr.val_dataset = [
            {'pixel_values': torch.randn(2, 4, 3, 16, 16), 'labels': torch.tensor(i % 2)}
            for i in range(4)
        ]

    @staticmethod
    def backbone(*args, **kwargs):
        config = VideoMAEConfig(image_size=16, patch_size=8, num_frames=4, tubelet_size=2,
                               hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=16)
        return VideoMAEModel(config)

    def test_both_models_train_predict_and_reload(self):
        with patch('videomae_local.models.VideoMAEModel.from_pretrained', side_effect=self.backbone):
            for name, cls in [('baseline', MeanPoolVideoMAEClassifier), ('proposed', EventLevelVideoUnderstandingModel)]:
                with self.subTest(model=name):
                    model = cls('tiny-local', num_labels=2, label_smoothing=.05)
                    trainer, metrics = tr.train_model(model, name)
                    self.assertIn('eval_top1_macro_f1', metrics)
                    pred = trainer.predict(tr.val_dataset)
                    self.assertEqual(pred.predictions.shape, (4, 2))
                    self.assertEqual(trainer.state.global_step, 4)
                    self.assertIsNotNone(tr.find_last_valid_checkpoint(str(Path(self.tmp.name) / name)))
                    state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
                    reloaded, _ = tr.train_model(cls('tiny-local', num_labels=2, label_smoothing=.05), name)
                    for k, v in reloaded.model.state_dict().items():
                        self.assertTrue(torch.equal(state[k], v), k)

    def test_video_decode_sampling_and_processor(self):
        import av
        import numpy as np
        from transformers import VideoMAEImageProcessor
        from videomae_local.models import VideoFolderDataset, VideoDataCollator
        root = Path(self.tmp.name)
        cls_dir = root / 'train' / 'dive'
        cls_dir.mkdir(parents=True)
        video = cls_dir / 'clip.mp4'
        with av.open(str(video), mode='w') as container:
            stream = container.add_stream('mpeg4', rate=8)
            stream.width = stream.height = 16
            stream.pix_fmt = 'yuv420p'
            for i in range(8):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 25, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        processor = VideoMAEImageProcessor(size={'shortest_edge': 16}, crop_size={'height': 16, 'width': 16})
        dataset = VideoFolderDataset(root, 'train', processor, {'dive': 0}, num_frames=4, num_clips=2, train=False)
        batch = VideoDataCollator()([dataset[0]])
        self.assertEqual(tuple(batch['pixel_values'].shape), (1, 2, 4, 3, 16, 16))
        self.assertEqual(batch['labels'].tolist(), [0])
        self.assertTrue(torch.isfinite(batch['pixel_values']).all())

    def test_run_exports_and_configuration_guard(self):
        import os
        from types import SimpleNamespace
        from transformers import VideoMAEImageProcessor
        root = Path(self.tmp.name)
        prepared = root / 'prepared'
        prepared.mkdir()
        (prepared / 'READY.json').write_text(json.dumps({'signature': 'test-split'}))
        (prepared / 'label_map.json').write_text(json.dumps({'label2id': {'a': 0, 'b': 1}}))
        class Samples(list):
            pass
        samples = Samples(tr.train_dataset)
        samples.samples = [(f'clip-{i}.mp4', i % 2) for i in range(len(samples))]
        options = SimpleNamespace(work_dir=root, batch_size=1, eval_batch_size=1, epochs=1,
                                  learning_rate=2e-5, seed=42, precision='fp32', patience=3,
                                  num_workers=0, model='both')
        processor = VideoMAEImageProcessor()
        with patch.dict(os.environ, {'MPLCONFIGDIR': str(root / 'mpl')}), \
             patch('videomae_local.models.VideoMAEModel.from_pretrained', side_effect=self.backbone), \
             patch.object(tr.VideoMAEImageProcessor, 'from_pretrained', return_value=processor), \
             patch.object(tr, 'VideoFolderDataset', return_value=samples), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=16 * 1024**3)):
            tr.run(options, prepared)
            for name in ('baseline', 'proposed'):
                dest = root / 'final_outputs' / name
                for filename in ('metrics.json', 'test_predictions.npz', 'test_samples.csv',
                                 'confusion_matrix.csv', 'confusion_matrix.png', 'classification_report.csv',
                                 'model/model.safetensors', 'model/config.json', 'model/preprocessor_config.json'):
                    self.assertTrue((dest / filename).is_file(), filename)
            self.assertTrue((root / 'final_outputs' / 'model_comparison.csv').is_file())
            options.epochs = 2
            with self.assertRaisesRegex(ValueError, 'Resume configuration changed'):
                tr.run(options, prepared)

    def test_resume_after_epoch(self):
        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                control.should_training_stop = True
        original_trainer = tr.Trainer
        def make_trainer(**kwargs):
            kwargs['callbacks'].append(StopAfterEpoch())
            return original_trainer(**kwargs)
        with patch('videomae_local.models.VideoMAEModel.from_pretrained', side_effect=self.backbone):
            with patch.object(tr, 'Trainer', side_effect=make_trainer):
                trainer, _ = tr.train_model(MeanPoolVideoMAEClassifier('tiny', 2), 'resume')
            self.assertEqual(trainer.state.global_step, 2)
            # Simulate interruption after the saved epoch, before successful completion marker.
            (Path(self.tmp.name) / 'resume' / 'TRAINING_DONE.json').unlink()
            resumed, _ = tr.train_model(MeanPoolVideoMAEClassifier('tiny', 2), 'resume')
            self.assertEqual(resumed.state.global_step, 4)
            self.assertEqual(resumed.state.epoch, 2)
            done = json.loads((Path(self.tmp.name) / 'resume' / 'TRAINING_DONE.json').read_text())
            self.assertEqual(done['global_step'], 4)


if __name__ == '__main__':
    unittest.main()
