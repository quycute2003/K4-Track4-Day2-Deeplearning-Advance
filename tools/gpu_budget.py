"""Measure one real DeepWeeds train/val epoch; extrapolate a GPU budget.

This diagnostic does not implement the lab pipeline or select models by quality.
It never evaluates test images. Import ML dependencies only in `measure`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

BACKBONES = ["resnet50", "resnext50_32x4d", "convnext_tiny",
             "deit_small_patch16_224", "efficientnet_b0"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fold(labels_dir, images_dir):
    splits = {}
    for split in ("train", "val", "test"):
        with (Path(labels_dir) / f"{split}_subset0.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        names = [r["Filename"] for r in rows]
        if not rows or len(names) != len(set(names)):
            raise ValueError(f"Empty split or duplicate filenames: {split}")
        for r in rows:
            if not 0 <= int(r["Label"]) < 9:
                raise ValueError(f"Invalid Label: {r}")
            if not (Path(images_dir) / r["Filename"]).is_file():
                raise ValueError(f"Missing image: {r['Filename']}")
        splits[split] = rows
    sets = {s: {r["Filename"] for r in rows} for s, rows in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Overlap: {a}/{b}")
    if len(set.union(*sets.values())) != 17509:
        raise ValueError("Fold 0 must contain 17,509 distinct filenames.")
    for split, rows in splits.items():
        expected = 0.6 if split == "train" else 0.2
        if abs(len(rows) / 17509 - expected) > 0.01:
            raise ValueError(f"Unexpected split size: {split}={len(rows)}")
    return splits


def measure(args):
    import numpy as np
    import torch
    import timm
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU in Colab/Kaggle before measuring.")
    splits = load_fold(args.labels_dir, args.images_dir)
    if args.batch_size > len(splits["train"]):
        raise ValueError("Batch size cannot exceed the train split size.")
    if len(set(args.backbones)) != len(args.backbones):
        raise ValueError("Backbone names must be distinct.")
    # Local Dataset classes can use Linux fork workers; Windows uses spawn.
    workers = 0 if platform.system() == "Windows" else args.workers

    class Images(Dataset):
        def __init__(self, rows, transform):
            self.rows, self.transform = rows, transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(Path(args.images_dir) / row["Filename"]) as img:
                x = self.transform(img.convert("RGB"))
            return x, int(row["Label"])

    report = {
        "purpose": "One-epoch timing only; no final lab results or test inference",
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "torch": torch.__version__, "timm": timm.__version__,
        "torchvision": torchvision.__version__, "seed": args.seed,
        "batch_size": args.batch_size, "img_size": args.img_size,
        "num_workers": workers, "amp": True,
        "planned_epochs": args.epochs,
        "split_sizes": {s: len(rows) for s, rows in splits.items()},
        "per_class": {s: [sum(int(r["Label"]) == c for r in rows) for c in range(9)]
                      for s, rows in splits.items()},
        "recipe": {"optimizer": "AdamW", "lr_backbone": 1e-4, "lr_head": 1e-3,
                   "weight_decay": 0.05, "norm_bias_decay": 0,
                   "loss": "CE", "init": "ImageNet finetune",
                   "scheduler": "1 epoch linear warmup, then cosine"},
        "models": [],
    }
    print("GPU:", report["gpu"], "| splits:", report["split_sizes"], flush=True)
    for name in args.backbones:
        # Same seed, batch size, recipe and split for every architecture.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        setup_start = time.perf_counter()
        kwargs = {"img_size": args.img_size} if name.startswith(("vit_", "deit_", "swin_")) else {}
        model = timm.create_model(name, pretrained=True, num_classes=9, **kwargs).cuda()
        cfg = dict(model.pretrained_cfg)
        norm = T.Normalize(cfg.get("mean", (0.485, 0.456, 0.406)),
                           cfg.get("std", (0.229, 0.224, 0.225)))
        interp = T.InterpolationMode.BICUBIC if cfg.get("interpolation") == "bicubic" else T.InterpolationMode.BILINEAR
        train_tf = T.Compose([T.RandomResizedCrop(args.img_size, interpolation=interp),
                              T.RandomHorizontalFlip(), T.ToTensor(), norm])
        val_tf = T.Compose([T.Resize(round(args.img_size * 256 / 224), interpolation=interp),
                            T.CenterCrop(args.img_size), T.ToTensor(), norm])
        loaders = {s: DataLoader(Images(splits[s], transform), batch_size=args.batch_size,
                                shuffle=(s == "train"), drop_last=(s == "train"),
                                num_workers=workers, pin_memory=True)
                   for s, transform in (("train", train_tf), ("val", val_tf))}
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        groups = {}
        for p in model.parameters():
            key = (1e-3 if id(p) in head_ids else 1e-4, 0.05 if p.ndim > 1 else 0.0)
            groups.setdefault(key, []).append(p)
        optimizer = torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": wd}
                                      for (lr, wd), p in groups.items()])
        steps = len(loaders["train"])
        def lr_factor(step):
            if step < steps:
                return (step + 1) / steps
            progress = min(1.0, (step - steps) / max(1, (args.epochs - 1) * steps))
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - setup_start
        torch.cuda.reset_peak_memory_stats()
        model.train()
        start = time.perf_counter()
        train_n = 0
        for x, y in loaders["train"]:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_n += len(y)
        torch.cuda.synchronize()
        train_s = time.perf_counter() - start
        model.eval()
        start = time.perf_counter()
        val_n = 0
        with torch.inference_mode():
            for x, y in loaders["val"]:
                x = x.cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                # Include CPU transfer used by the actual evaluation pipeline.
                logits.float().cpu().numpy()
                val_n += len(y)
        torch.cuda.synchronize()
        val_s = time.perf_counter() - start
        row = {"backbone": name, "pretrained_cfg": cfg,
               "train_images_seen": train_n, "val_images_seen": val_n,
               "setup_s": setup_s, "train_s": train_s, "val_s": val_s,
               "epoch_s": train_s + val_s,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        report["models"].append(row)
        write_json(args.out, report)  # Preserve completed models if the next one fails.
        print(f"{name}: train={train_s:.1f}s, val={val_s:.1f}s, epoch={train_s + val_s:.1f}s", flush=True)
        del model, optimizer, scheduler, scaler, groups, loaders, x, y, loss, logits
        gc.collect()
        torch.cuda.empty_cache()
    print("Saved:", args.out)


def plan(args):
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    rows = profile["models"]
    times = {r["backbone"]: float(r["epoch_s"]) for r in rows}
    if len(times) < 5:
        raise ValueError("Measure at least five distinct backbones before planning.")
    if any(not math.isfinite(t) or t <= 0 for t in times.values()):
        raise ValueError("Epoch timings must be finite and positive.")
    if not set(args.selected) <= times.keys():
        raise ValueError("Every ablation backbone must have a measured epoch.")
    if len(set(args.selected)) != len(args.selected):
        raise ValueError("Ablation backbones must be distinct.")
    final = args.final_backbone or args.selected[0]
    if final not in args.selected:
        raise ValueError("Final backbone must belong to --selected.")
    epoch_h = lambda names: sum(times[n] for n in names) * args.epochs / 3600
    # 7 fresh runs per ablation backbone: scratch, frozen, color, CutMix,
    # label smoothing, focal, plus one combination. T00 is the B-stage recipe.
    stages = [
        ("B: backbone", len(times), epoch_h(list(times))),
        ("T: 6 ablations + 1 combination / backbone", 7 * len(args.selected),
         7 * epoch_h(args.selected)),
        ("F: final 3 seeds + baseline 3 seeds", 6, 6 * epoch_h([final])),
    ]
    profiling_h = sum(float(r["setup_s"]) + times[r["backbone"]] for r in rows) / 3600
    training_h = sum(s[2] for s in stages)
    future_h = (training_h + args.extra_hours) * (1 + args.reserve)
    total_h = profiling_h + future_h
    lines = ["# Kế hoạch ngân sách GPU", "",
             "Ước lượng từ một epoch train + val; chưa phải thời gian chạy toàn bộ thí nghiệm.",
             "Chọn backbone trong kế hoạch chỉ để tính ngân sách. Quyết định chất lượng phải dựa trên macro-F1 val.",
             "", f"GPU đã đo: {profile['gpu']}. Batch: {profile['batch_size']}; kích thước: {profile['img_size']}; AMP: bật.",
             f"Epoch mỗi lần chạy: {args.epochs}. Backbone ablation dự kiến: {', '.join(args.selected)}.",
             "", "| Bước | Lần huấn luyện | Giờ GPU ước lượng |", "|---|---:|---:|"]
    lines += [f"| {label} | {count} | {hours:.2f} |" for label, count, hours in stages]
    lines += ["", f"Tổng số lần huấn luyện mới: {sum(s[1] for s in stages)} (ngoài profiling).",
              "Không tái sử dụng profiling 1 epoch làm kết quả backbone 10–15 epoch.",
              f"Profiling đã dùng: {profiling_h:.2f} giờ (gồm tải/khởi tạo model).",
              f"Khoản thêm cho EDA, suy luận, test, benchmark: {args.extra_hours:.2f} giờ, là khoản dự trù chưa đo.",
              f"Dự phòng: {args.reserve:.0%}; GPU còn cần: {future_h:.2f} giờ; tổng kể cả profiling: {total_h:.2f} giờ."]
    if args.gpu_hours is None:
        lines += ["", "**Chưa chốt kế hoạch:** chưa nhập hạn mức GPU. Chạy lại `plan --gpu-hours ...` sau khi kiểm tra tài khoản."]
    else:
        fits = total_h <= args.gpu_hours
        lines += ["", f"Hạn mức tổng đã nhập: {args.gpu_hours:.2f} giờ; "
                  + ("**ước lượng vừa ngân sách**." if fits else "**ước lượng vượt ngân sách**."),
                  "Hạn mức này gồm profiling; dùng số giờ cùng một chu kỳ hạn mức."]
        if not fits:
            lines += ["Giảm 15 → 10 epoch, ablation trên 1 backbone, giữ 3 seed cho final và baseline; "
                      "nếu hạ độ phân giải/batch hoặc đổi GPU thì đo lại toàn bộ backbone."]
    lines += ["", "T00 dùng lại công thức của B-stage trên backbone được chọn; vòng cuối vẫn tính 3 lần baseline mới.",
              "Seed sàng lọc: 0. Seed vòng cuối: 0, 1, 2. Mỗi ablation chỉ đổi một yếu tố.",
              "Các lần sàng lọc không đánh giá test. Sau khi chốt trên val, chạy test một lần cho mỗi cấu hình/seed.",
              "Cần đo lại nếu pipeline cuối khác profiler; chi phí CutMix/focal/EMA có thể khác công thức nền."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def nonnegative(value):
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise argparse.ArgumentTypeError("Must be finite and nonnegative")
    return n


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    m = commands.add_parser("measure")
    m.add_argument("--images-dir", default="data/images")
    m.add_argument("--labels-dir", default="data/labels")
    m.add_argument("--backbones", nargs="+", default=BACKBONES)
    m.add_argument("--batch-size", type=positive_int, default=32)
    m.add_argument("--img-size", type=positive_int, default=224)
    m.add_argument("--workers", type=positive_int, default=2)
    m.add_argument("--epochs", type=positive_int, default=10)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default="runs/gpu_budget/profile.json")
    m.set_defaults(func=measure)
    p = commands.add_parser("plan")
    p.add_argument("--profile", default="runs/gpu_budget/profile.json")
    p.add_argument("--selected", nargs="+", required=True)
    p.add_argument("--final-backbone")
    p.add_argument("--epochs", type=positive_int, default=10)
    p.add_argument("--gpu-hours", type=nonnegative)
    p.add_argument("--reserve", type=nonnegative, default=0.25)
    p.add_argument("--extra-hours", type=nonnegative, default=1.0)
    p.add_argument("--out", default="runs/gpu_budget/plan.md")
    p.set_defaults(func=plan)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
