"""Regression checks for recipe locking, reuse and resumable screening."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_implementation import train

step1 = importlib.import_module('deepweeds_lab.step1')
ROOT = Path(__file__).resolve().parent.parent


class TestStep1(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Tiny metadata fixture, independent of downloaded project results.
        cfg = train.Config(exp_id='B01', epochs=10, batch_size=32)
        from dataclasses import asdict
        self.reference = {'config': asdict(cfg), 'gpu': 'Tesla T4',
                          'versions': {'torch': 'fixture', 'torchvision': 'fixture', 'timm': 'fixture'},
                          'pretrained_cfg': {'architecture': 'resnet50', 'tag': 'fixture'}}
        folder = self.root / 'runs/B01/seed0'
        folder.mkdir(parents=True)
        (folder / 'config.json').write_text(json.dumps(self.reference))
        summary = dict(exp_id='B01', seed=0, backbone='resnet50', best_epoch=10,
                       macro_f1_val=.8, top1_val=.9, params_m=23, gmacs=4,
                       test_evaluated=False)
        (folder / 'summary.json').write_text(json.dumps(summary))
        (folder / 'history.csv').write_text('epoch,train_seconds,epoch_seconds\n'+
                                          '\n'.join(f'{i},1,2' for i in range(1,11)))
        (self.root / 'curves').mkdir()
        (self.root / 'curves/B01_resnet50_seed0.png').touch()

    def tearDown(self):
        self.temp.cleanup()

    def test_recipe_locked_except_backbone_and_paths(self):
        cfg = step1.make_config(self.reference, self.root, 'images', 'labels', 'B04',
                                'deit_small_patch16_224')
        self.assertEqual((cfg.epochs, cfg.batch_size, cfg.seed), (10, 32, 0))
        from dataclasses import asdict
        values = asdict(cfg)
        for field, value in [('lr_head', .0001), ('epochs', 12), ('seed', 1),
                             ('save_test_predictions', True)]:
            changed = dict(values, **{field: value})
            with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
                step1.assert_same_recipe(self.reference['config'], changed)

    def test_convnext_screening_uses_explicit_imagenet1k_tag(self):
        with mock.patch('timm.create_model', return_value=mock.MagicMock()) as create:
            step1.models.build_model('convnext_tiny', init='finetune')
        self.assertEqual(create.call_args.args[0], 'convnext_tiny.fb_in1k')
        self.assertTrue(create.call_args.kwargs['pretrained'])

    def test_partial_results_keep_unmeasured_latency_and_runs_missing(self):
        rows = step1.collect_results(self.root)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]['macro_f1_val'], .8)
        self.assertIsNone(rows[0]['latency_ms_batch1_fp32'])
        for row in rows[1:]:
            self.assertEqual(row['status'], 'Chưa chạy')
            self.assertIsNone(row['macro_f1_val'])
        with self.assertRaisesRegex(ValueError, 'all five'):
            step1.write_report(self.root)
        self.assertFalse((self.root / 'step1/selection.json').exists())

    def test_completed_b01_never_retrained(self):
        with mock.patch.object(step1, 'assert_environment'), \
             mock.patch.object(step1, 'measure_latency') as latency, \
             mock.patch.object(train, 'run') as runner:
            step1.run_backbone(self.root, 'images', 'labels', 'B01')
        runner.assert_not_called()
        self.assertEqual(latency.call_args.args[1].backbone, 'resnet50')

    def test_resume_keeps_backbone_and_rejects_changed_recipe(self):
        folder = self.root / 'runs/B02/seed0'
        folder.mkdir(parents=True)
        record = json.loads(json.dumps(self.reference))
        record['config'].update(exp_id='B02', backbone='resnext50_32x4d')
        (folder / 'config.json').write_text(json.dumps(record))
        (folder / 'last.pt').touch()
        with mock.patch.object(step1, 'assert_environment'), \
             mock.patch.object(train, 'run', side_effect=RuntimeError('fixture stop')) as runner:
            with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                step1.run_backbone(self.root, 'images', 'labels', 'B02')
        self.assertTrue(runner.call_args.args[0].resume)
        self.assertFalse(runner.call_args.args[0].save_test_predictions)
        record['config']['epochs'] = 12
        (folder / 'config.json').write_text(json.dumps(record))
        with mock.patch.object(step1, 'assert_environment'), mock.patch.object(train, 'run') as runner:
            with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
                step1.run_backbone(self.root, 'images', 'labels', 'B02')
        runner.assert_not_called()

    def test_selection_requires_real_complete_rows_and_uses_val_only(self):
        for i, (exp_id, backbone, family) in enumerate(step1.BACKBONES):
            folder = self.root / 'runs' / exp_id / 'seed0'
            folder.mkdir(parents=True, exist_ok=True)
            record = json.loads(json.dumps(self.reference))
            record['config'].update(exp_id=exp_id, backbone=backbone)
            (folder / 'config.json').write_text(json.dumps(record))
            summary = dict(exp_id=exp_id, seed=0, backbone=backbone, best_epoch=10,
                           macro_f1_val=.8 + .01*i, top1_val=.9, params_m=23,
                           gmacs=4+i, test_evaluated=False)
            (folder / 'summary.json').write_text(json.dumps(summary))
            (folder / 'history.csv').write_text(
                'epoch,train_seconds,epoch_seconds,train_loss,val_loss,val_macro_f1\n'+
                '\n'.join(f'{j},1,2,{1/j},{1/j},{.8+.01*i}' for j in range(1,11)))
            latency = dict(latency_ms=10-i, gpu='Tesla T4', torch='fixture')
            (folder / 'latency_preliminary.json').write_text(json.dumps(latency))
            (self.root / 'curves' / f'{exp_id}_{backbone}_seed0.png').touch()
        chosen = step1.write_report(self.root)
        self.assertEqual(chosen['selected_exp_ids'], ['B05'])
        self.assertFalse(chosen['test_used'])
        self.assertIn('một seed', (self.root / 'step1/step1_report.md').read_text(encoding='utf-8').lower())
