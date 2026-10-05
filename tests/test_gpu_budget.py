"""Budget arithmetic tests; these synthetic timings are test fixtures only."""
import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.gpu_budget import BACKBONES, plan


class TestBudget(unittest.TestCase):
    def run_plan(self, selected, gpu_hours=None, epoch_s=100):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"gpu": "test fixture", "batch_size": 32, "img_size": 224,
                       "models": [{"backbone": n, "epoch_s": epoch_s, "setup_s": 0}
                                  for n in BACKBONES]}
            (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            args = Namespace(profile=root / "profile.json", selected=selected,
                             final_backbone=None, epochs=10, gpu_hours=gpu_hours,
                             reserve=0.25, extra_hours=1, out=root / "plan.md")
            with contextlib.redirect_stdout(io.StringIO()):
                plan(args)
            return (root / "plan.md").read_text(encoding="utf-8")

    def test_one_backbone_includes_both_three_seed_groups(self):
        result = self.run_plan([BACKBONES[0]], gpu_hours=8)
        # 18*10*100/3600=5 h training; (5+1)*1.25 + 500/3600=7.64 h total.
        self.assertIn("huấn luyện mới: 18", result)
        self.assertIn("tổng kể cả profiling: 7.64", result)
        self.assertIn("ước lượng vừa ngân sách", result)

    def test_two_ablation_backbones_cost_more(self):
        result = self.run_plan(BACKBONES[:2], gpu_hours=8)
        self.assertIn("huấn luyện mới: 25", result)
        self.assertIn("ước lượng vượt ngân sách", result)

    def test_missing_quota_is_not_confirmed(self):
        self.assertIn("Chưa chốt kế hoạch", self.run_plan([BACKBONES[0]]))

    def test_invalid_measurements_and_selections_rejected(self):
        for names, seconds in (([BACKBONES[0]], 0), ([BACKBONES[0]], float("nan")),
                               (["not_measured"], 100), ([BACKBONES[0]] * 2, 100)):
            with self.subTest(names=names, seconds=seconds), self.assertRaises(ValueError):
                self.run_plan(names, epoch_s=seconds)


if __name__ == "__main__":
    unittest.main()
