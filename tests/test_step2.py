"""Guard against recipe drift, val misalignment and accidental retraining."""
import importlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
from test_implementation import train

step2 = importlib.import_module('deepweeds_lab.step2')


class TestStep2(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        cfg = train.Config(exp_id='B03', backbone='convnext_tiny', epochs=10, batch_size=32)
        self.reference = dict(config=asdict(cfg), gpu='Tesla T4',
                              versions=dict(torch='fixture', torchvision='fixture', timm='fixture'),
                              pretrained_cfg=dict(tag='fb_in1k'))
        folder = self.root / 'runs/B03/seed0'
        folder.mkdir(parents=True)
        train.write_json(folder / 'config.json', self.reference)
        self.y = np.tile(np.arange(9), 3)
        self.names = [f'{i}.jpg' for i in range(len(self.y))]
        self.write_run('B03', self.y)

    def tearDown(self):
        self.temp.cleanup()

    def write_run(self, exp_id, predicted):
        folder = self.root / 'runs' / exp_id / 'seed0'
        folder.mkdir(parents=True, exist_ok=True)
        overrides = {} if exp_id == 'B03' else step2.spec(self.root, exp_id)[2]
        record = json.loads(json.dumps(self.reference))
        record['config'].update(exp_id=exp_id, **overrides)
        train.write_json(folder / 'config.json', record)
        logits = np.zeros((len(self.y), 9))
        logits[np.arange(len(self.y)), predicted] = 3
        probs = train.probabilities(logits)
        metrics = train.ev.compute_metrics(self.y, predicted, probs)
        train.write_json(folder / 'summary.json', dict(exp_id=exp_id, seed=0, best_epoch=1,
            test_evaluated=False, macro_f1_val=metrics['macro_f1'], top1_val=metrics['top1'], ece_val=metrics['ece']))
        np.save(folder / 'val_logits.npy', logits)
        np.save(folder / 'val_labels.npy', self.y)
        train.write_json(folder / 'val_filenames.json', self.names)
        prediction = self.root / 'predictions' / f'{exp_id}_seed0_val.csv'
        prediction.parent.mkdir(exist_ok=True)
        train.ev.save_predictions(str(prediction), self.names, self.y, probs)
        (folder / 'history.csv').write_text('epoch,train_seconds,epoch_seconds,val_loss,train_loss,val_macro_f1,val_top1,lr\n'+
            '\n'.join(f'{i},1,2,0.5,0.5,{metrics["macro_f1"]},{metrics["top1"]},0.001' for i in range(1, 11)))
        (self.root / 'curves').mkdir(exist_ok=True)
        (self.root / f'curves/{exp_id}_convnext_tiny_seed0.png').touch()

    def test_unrun_metrics_blank_and_no_final_selection(self):
        rows = step2.collect_results(self.root)
        self.assertEqual(rows[0]['macro_f1_val'], 1)
        self.assertTrue(all(row['macro_f1_val'] is None for row in rows[1:]))
        with self.assertRaisesRegex(ValueError, 'Finish all'):
            step2.write_report(self.root)
        self.assertFalse((self.root / 'step2/selection.json').exists())

    def test_backbone_seed_lr_and_test_cannot_drift(self):
        cfg = step2.make_config(self.root, 'images', 'labels', 'T04')
        values = asdict(cfg)
        step2.assert_controlled(self.reference['config'], values, {'mix':'cutmix'})
        for field, value in [('backbone','resnet50'), ('seed',1), ('lr_head',.5),
                             ('save_test_predictions',True), ('epochs',12)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Uncontrolled'):
                step2.assert_controlled(self.reference['config'], dict(values, **{field:value}), {'mix':'cutmix'})

    def test_t00_and_completed_trial_never_train(self):
        self.write_run('T01', self.y)
        with mock.patch.object(step2.step1, 'assert_environment'), mock.patch.object(train, 'run') as runner:
            step2.run_recipe(self.root, 'images', 'labels', 'T00')
            step2.run_recipe(self.root, 'images', 'labels', 'T01')
        runner.assert_not_called()

    def test_interrupted_run_resumes_its_own_recipe(self):
        folder = self.root / 'runs/T04/seed0'
        folder.mkdir(parents=True)
        record = json.loads(json.dumps(self.reference))
        record['config'].update(exp_id='T04', mix='cutmix')
        train.write_json(folder / 'config.json', record)
        (folder / 'last.pt').touch()
        with mock.patch.object(step2.step1, 'assert_environment'), \
             mock.patch.object(train, 'run', side_effect=RuntimeError('stop')) as runner:
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                step2.run_recipe(self.root, 'images', 'labels', 'T04')
        cfg = runner.call_args.args[0]
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.mix, 'cutmix')
        self.assertFalse(cfg.save_test_predictions)

    def test_changed_checkpoint_recipe_rejected_before_training(self):
        self.write_run('T01', self.y)
        path = self.root / 'runs/T01/seed0/config.json'
        record = step2.step1.read_json(path)
        record['config']['lr_backbone'] = .2
        train.write_json(path, record)
        with mock.patch.object(step2.step1, 'assert_environment'), mock.patch.object(train, 'run') as runner:
            with self.assertRaisesRegex(ValueError, 'Uncontrolled'):
                step2.run_recipe(self.root, 'images', 'labels', 'T01')
        runner.assert_not_called()

    def test_val_alignment_and_saved_metrics_are_verified(self):
        self.write_run('T03', self.y)
        path = self.root / 'runs/T03/seed0/val_filenames.json'
        train.write_json(path, list(reversed(self.names)))
        with self.assertRaisesRegex(ValueError, 'alignment'):
            step2.collect_results(self.root)

    def test_combination_requires_results_and_is_fixed_before_run(self):
        with self.assertRaisesRegex(ValueError, 'completed'):
            step2.plan_combination(self.root)
        for exp in ('T03','T04','T05','T06'):
            self.write_run(exp, (self.y+1)%9)
        plan = step2.plan_combination(self.root)
        self.assertEqual(plan['selected_components'], ['T03','T05'])
        self.assertEqual(plan['overrides'], dict(aug='color',loss='ls',label_smoothing=.1))
        self.assertFalse(plan['test_used'])
        self.assertEqual(step2.plan_combination(self.root), plan)
        self.write_run('T04', self.y)
        with self.assertRaisesRegex(ValueError, 'choices changed'):
            step2.plan_combination(self.root)

    def test_baseline_wins_when_alternatives_lose_and_on_ties(self):
        for exp in step2.EXPERIMENTS:
            self.write_run(exp, (self.y+1)%9)
        step2.plan_combination(self.root)
        self.write_run('T07', self.y)
        selection = step2.write_report(self.root)
        self.assertEqual(selection['selected_recipe'], 'T00')
        self.assertEqual(selection['source_exp_id'], 'B03')
        self.assertFalse(selection['test_used'])

    def test_paired_bootstrap_identical_predictions_have_zero_delta(self):
        ci = step2.paired_delta_ci(self.y, self.y, self.y, repeats=20)
        self.assertEqual((ci['lo_pp'], ci['hi_pp'], ci['bootstrap_std_pp']), (0,0,0))

    def test_checkpoint_zip_keeps_experiment_paths_and_requires_all_runs(self):
        import zipfile
        archive = self.root / 'checkpoints.zip'
        with self.assertRaises(FileNotFoundError):
            step2.export_checkpoints(self.root, archive)
        for exp in [*step2.EXPERIMENTS,'T07']:
            folder = self.root / f'runs/{exp}/seed0'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'best.pt').write_bytes(exp.encode())
            (folder / 'last.pt').write_bytes(b'not included')
        step2.export_checkpoints(self.root, archive)
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(set(bundle.namelist()), {f'runs/{exp}/seed0/best.pt' for exp in [*step2.EXPERIMENTS,'T07']})
            self.assertEqual(bundle.read('runs/T03/seed0/best.pt'), b'T03')

    def test_resume_rejects_environment_change_before_training(self):
        self.write_run('T01', self.y)
        path = self.root / 'runs/T01/seed0/config.json'
        record = step2.step1.read_json(path)
        record['gpu'] = 'another GPU'
        train.write_json(path, record)
        with mock.patch.object(step2.step1, 'assert_environment'), mock.patch.object(train, 'run') as runner:
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                step2.run_recipe(self.root, 'images', 'labels', 'T01')
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
