"""CPU tests for the completed implementation; synthetic images are fixtures only."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("deepweeds_lab", ROOT / "code/_package.py",
                                             submodule_search_locations=[str(ROOT / "code")])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from deepweeds_lab import dataset, losses, model, train, step0  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                      nn.Dropout(0.1), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4, 9)
        self.pretrained_cfg = {}

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class TestLosses(unittest.TestCase):
    def test_focal_zero_matches_ce_and_gradients(self):
        torch.manual_seed(7)
        logits = torch.randn(11, 9, requires_grad=True)
        targets = torch.arange(11) % 9
        focal = losses.FocalLoss(gamma=0)(logits, targets)
        ce = nn.functional.cross_entropy(logits, targets)
        self.assertLess(float((focal - ce).detach().abs()), 1e-6)
        a = torch.autograd.grad(focal, logits, retain_graph=True)[0]
        b = torch.autograd.grad(ce, logits)[0]
        torch.testing.assert_close(a, b)

    def test_weighted_focal_zero_and_smoothing_zero_match_ce(self):
        x, y = torch.randn(12, 9), torch.arange(12) % 9
        weight = torch.arange(1, 10).float()
        torch.testing.assert_close(losses.FocalLoss(0, weight)(x, y),
                                   nn.functional.cross_entropy(x, y, weight=weight))
        torch.testing.assert_close(losses.LabelSmoothingCE(0)(x, y),
                                   nn.functional.cross_entropy(x, y))

    def test_cutmix_clipped_area_labels_and_original_image(self):
        x = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
        y = torch.tensor([2, 7])
        original = x.clone()
        with mock.patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.2)), \
             mock.patch("torch.randperm", return_value=torch.tensor([1, 0])), \
             mock.patch("torch.randint", return_value=torch.tensor([0])):
            mixed, (a, b, lam) = losses.mix_batch(x, y)
        changed_area = int((mixed[0, 0] != x[0, 0]).sum())
        self.assertGreater(changed_area, 0)
        self.assertAlmostEqual(lam, 1 - changed_area / 64)
        self.assertNotAlmostEqual(lam, 0.2)  # Clipping really changed the coefficient.
        torch.testing.assert_close(a, y)
        torch.testing.assert_close(b, y.flip(0))
        torch.testing.assert_close(x, original)
        logits = torch.randn(2, 9)
        criterion = nn.CrossEntropyLoss()
        torch.testing.assert_close(losses.mixed_loss(criterion, logits, (a, b, lam)),
                                   lam * criterion(logits, a) + (1 - lam) * criterion(logits, b))

    def test_mixup_and_disable(self):
        x, y = torch.randn(2, 3, 4, 4), torch.tensor([0, 1])
        with mock.patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.25)), \
             mock.patch("torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, (_, b, lam) = losses.mix_batch(x, y, mode="mixup")
        torch.testing.assert_close(mixed, 0.25 * x + 0.75 * x.flip(0))
        torch.testing.assert_close(b, y.flip(0))
        self.assertEqual(lam, 0.25)
        untouched, targets = losses.mix_batch(x, y, alpha=0)
        self.assertIs(untouched, x)
        self.assertEqual(targets[2], 1)

    def test_class_weights_effective_samples_and_validation(self):
        counts = np.arange(1, 10) * 10
        for beta in (0, 0.9999):
            w = losses.class_weights(counts, beta)
            self.assertAlmostEqual(float(w.mean()), 1, places=6)
            self.assertTrue(torch.all(w[:-1] > w[1:]))
        with self.assertRaises(ValueError):
            losses.class_weights([0] * 9)


class TestModel(unittest.TestCase):
    def test_groups_cover_each_trainable_parameter_once_and_skip_bias_decay(self):
        net = TinyModel()
        groups = model.param_groups(net, 1e-4, 1e-3, 0.05)
        params = [p for group in groups for p in group["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in net.parameters()})
        head_ids = {id(p) for p in net.head.parameters()}
        for group in groups:
            for p in group["params"]:
                self.assertEqual(group["lr"], 1e-3 if id(p) in head_ids else 1e-4)
                self.assertEqual(group["weight_decay"], 0 if p.ndim <= 1 else 0.05)

    def test_frozen_features_keep_bn_stats_after_train_and_forward(self):
        net = TinyModel()
        model.freeze_backbone(net)
        net.train()
        bn = net.features[1]
        before = bn.running_mean.clone()
        self.assertFalse(bn.training)
        self.assertFalse(net.features[3].training)
        self.assertTrue(net.head.training)
        net(torch.randn(3, 3, 8, 8)).sum().backward()
        torch.testing.assert_close(bn.running_mean, before)
        self.assertTrue(all(p.grad is None for p in net.features.parameters()))
        self.assertTrue(all(p.grad is not None for p in net.head.parameters()))
        net.eval()
        self.assertFalse(net.head.training)

    def test_ema_averages_parameters_and_copies_bn_buffers(self):
        net = TinyModel()
        ema = train.EMA(net, 0.5)
        before = net.head.weight.detach().clone()
        with torch.no_grad():
            net.head.weight.add_(2)
            net.features[1].running_mean.fill_(3)
            net.features[1].num_batches_tracked.fill_(5)
        ema.update(net)
        torch.testing.assert_close(ema.module.head.weight, before + 1)
        torch.testing.assert_close(ema.module.features[1].running_mean, torch.full((4,), 3.))
        self.assertEqual(int(ema.module.features[1].num_batches_tracked), 5)

    def test_fvcore_counts_and_restores_mode(self):
        net = TinyModel().train()
        self.assertGreater(model.count_gmacs(net, 8), 0)
        self.assertTrue(net.training)
        self.assertTrue(net.features[1].training)
        self.assertIn("unsupported_ops", net.gmac_report)


class TestDataAndRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images, self.labels = self.root / "images", self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        self.frames = []
        offset = 0
        for split, n in (("train", 27), ("val", 9), ("test", 9)):
            rows = []
            for i in range(n):
                name = f"fixture{offset + i:03d}.jpg"
                label = i % 9
                color = (label * 25, (label * 61) % 256, (label * 101) % 256)
                Image.new("RGB", (20, 20), color).save(self.images / name)
                # Match the official subset CSV schema (no Species column).
                rows.append({"Filename": name, "Label": label})
            offset += n
            df = pd.DataFrame(rows)
            df.to_csv(self.labels / f"{split}_subset0.csv", index=False)
            self.frames.append(df)

    def tearDown(self):
        self.tmp.cleanup()

    def test_official_two_column_splits_load_without_modifying_csvs(self):
        before = {p.name: p.read_bytes() for p in self.labels.glob("*.csv")}
        frames = dataset.load_split(self.labels)
        for actual, expected in zip(frames, self.frames):
            pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(list(actual.columns), ["Filename", "Label"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.labels.glob("*.csv")})
        for column in ("Filename", "Label"):
            with self.assertRaisesRegex(ValueError, "required columns"):
                dataset._validate_frame(frames[0].drop(columns=column), "train")
        invalid = frames[0].copy()
        invalid.loc[0, "Label"] = 9
        with self.assertRaisesRegex(ValueError, "integers 0..8"):
            dataset._validate_frame(invalid, "train")

    def test_split_success_and_overlap_missing_duplicate_failures(self):
        with contextlib.redirect_stdout(io.StringIO()):
            info = dataset.check_split(*self.frames, self.images, expected_total=45)
        self.assertEqual(info["n"], {"train": 27, "val": 9, "test": 9})
        self.assertEqual(info["union"], 45)
        overlapping = self.frames[1].copy()
        overlapping.loc[0, "Filename"] = self.frames[0].iloc[0].Filename
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            dataset.check_split(self.frames[0], overlapping, self.frames[2], self.images, expected_total=45)
        duplicated = self.frames[0].copy()
        duplicated.loc[1, "Filename"] = duplicated.loc[0, "Filename"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dataset.check_split(duplicated, *self.frames[1:], self.images, expected_total=45)
        (self.images / self.frames[2].iloc[0].Filename).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            dataset.check_split(*self.frames, self.images, expected_total=45)

    def test_loader_reproducibility_eval_order_and_shapes(self):
        def sample_batch():
            train.set_seed(0)
            loader = dataset.make_loader(self.frames[0], self.images,
                                          dataset.build_transforms(True, 16), 9, True,
                                          num_workers=0, seed=3)
            return next(iter(loader))
        x1, y1, names1 = sample_batch()
        x2, y2, names2 = sample_batch()
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)
        self.assertEqual(names1, names2)
        tf = dataset.build_transforms(False, 16)
        loader = dataset.make_loader(self.frames[1], self.images, tf, 4, False, num_workers=0)
        net = TinyModel().train()
        running_mean = net.features[1].running_mean.clone()
        names, labels, logits, _ = train.evaluate(net, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertEqual(names, self.frames[1].Filename.tolist())
        np.testing.assert_array_equal(labels, self.frames[1].Label)
        self.assertEqual(logits.shape, (9, 9))
        self.assertFalse(net.training)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        torch.testing.assert_close(net.features[1].running_mean, running_mean)

    def test_eda_preserves_fold_labels_and_reports_master_discrepancy(self):
        reference = pd.concat(self.frames, ignore_index=True)
        reference["Species"] = reference.Label.map(dict(enumerate(dataset.CLASS_NAMES)))
        reference.loc[0, "Label"] = 1
        reference.loc[0, "Species"] = dataset.CLASS_NAMES[1]
        reference.to_csv(self.labels / "labels.csv", index=False)
        before = {p.name: p.read_bytes() for p in self.labels.glob("*.csv")}
        original_check, original_open = dataset.check_split, Image.open
        test_names = set(self.frames[2].Filename)
        def guarded_open(path, *a, **kw):
            self.assertNotIn(Path(path).name, test_names)
            return original_open(path, *a, **kw)
        with mock.patch.object(dataset, "check_split", side_effect=lambda *a: original_check(*a, expected_total=45)), \
             mock.patch.object(Image, "open", side_effect=guarded_open), \
             contextlib.redirect_stdout(io.StringIO()):
            eda = step0.analyze_data(self.images, self.labels, self.root / "eda")
        self.assertTrue(eda["reference_filenames_match"])
        self.assertFalse(eda["reference_labels_match"])
        self.assertEqual(len(eda["label_discrepancies"]), 1)
        self.assertEqual(eda["label_discrepancies"][0]["fold0_Label"], 0)
        self.assertEqual(eda["label_discrepancies"][0]["labels_csv_Label"], 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.labels.glob("*.csv")})
        for name in ("class_distribution.png", "samples_3_per_class.png", "cutmix_examples.png"):
            self.assertTrue((self.root / "eda" / name).is_file())

    def _config(self):
        return train.Config(exp_id="FIXTURE", init="scratch", epochs=2, batch_size=9,
                            img_size=16, num_workers=0, device="cpu", amp=False,
                            images_dir=str(self.images), labels_dir=str(self.labels),
                            out_dir=str(self.root / "runs"), pred_dir=str(self.root / "predictions"),
                            curves_dir=str(self.root / "curves"))

    def test_run_writes_real_contract_artifacts_and_never_decodes_test(self):
        original_check = dataset.check_split
        original_open = Image.open
        test_names = set(self.frames[2].Filename)
        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, test_names, "Test image decoded during screening")
            return original_open(path, *args, **kwargs)
        real_metrics = train.compute_metrics
        def tied_metrics(*args, **kwargs):
            metrics = real_metrics(*args, **kwargs)
            metrics["macro_f1"] = 0.25  # Deterministic tie fixture, not a lab result.
            return metrics
        cfg = self._config()
        with mock.patch.object(model, "build_model", side_effect=lambda *a, **k: TinyModel()), \
             mock.patch.object(dataset, "check_split", side_effect=lambda *a: original_check(*a, expected_total=45)), \
             mock.patch.object(train, "compute_metrics", side_effect=tied_metrics), \
             mock.patch.object(Image, "open", side_effect=guarded_open), \
             mock.patch.object(dataset, "make_loader", wraps=dataset.make_loader) as loaders, \
             contextlib.redirect_stdout(io.StringIO()):
            result = train.run(cfg)
            self.assertEqual(loaders.call_count, 2)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["best_epoch"], 1)
        folder = train.run_dir(cfg)
        for name in ("history.csv", "config.json", "best.pt", "last.pt", "val_logits.npy", "summary.json"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertTrue(Path(result["curve"]).is_file())
        self.assertEqual(len(pd.read_csv(folder / "history.csv")), 2)
        prediction = train.ev.read_pred(str(train.pred_path(cfg, "val")))
        train.ev.check_against_csv(prediction, str(self.labels / "val_subset0.csv"), "val")
        self.assertFalse(train.pred_path(cfg, "test").exists())
        with self.assertRaises(FileExistsError):
            train.run(cfg)

    def test_resume_continues_after_an_interrupted_epoch(self):
        cfg = self._config()
        original_check, original_epoch = dataset.check_split, train.train_one_epoch
        counter = [0]
        def interrupted_epoch(*a, **kw):
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError("simulated interruption")
            return original_epoch(*a, **kw)
        with mock.patch.object(model, "build_model", side_effect=lambda *a, **k: TinyModel()), \
             mock.patch.object(dataset, "check_split", side_effect=lambda *a: original_check(*a, expected_total=45)), \
             contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(train, "train_one_epoch", side_effect=interrupted_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train.run(cfg)
            self.assertEqual(len(pd.read_csv(train.run_dir(cfg) / "history.csv")), 1)
            cfg.resume = True
            train.run(cfg)
        self.assertEqual(len(pd.read_csv(train.run_dir(cfg) / "history.csv")), 2)


class TestConfigAndScheduler(unittest.TestCase):
    def test_cli_types_none_and_invalid_fields(self):
        self.assertEqual(train.parse_overrides(["seed=3", "amp=false", "ema_decay=none", "lr_head=0.001"]),
                         {"seed": 3, "amp": False, "ema_decay": None, "lr_head": 0.001})
        for pairs in (["unknown=1"], ["amp=yes"], ["epochs=x"], ["lr_head=nan"], ["seed"]):
            with self.assertRaises(ValueError):
                train.parse_overrides(pairs)
        self.assertFalse(train.Config().save_test_predictions)

    def test_warmup_cosine_reaches_zero(self):
        p = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([p], lr=1)
        scheduler = train.build_scheduler(optimizer, train.Config(epochs=3, warmup_epochs=1), 4)
        values = [optimizer.param_groups[0]["lr"]]
        for _ in range(12):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(values[:4], [0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(values[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(values[4:], values[5:])))


if __name__ == "__main__":
    unittest.main()
