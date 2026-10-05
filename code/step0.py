"""Real-data EDA, diagnostic checks and packaging for the Step 0 notebook."""
from __future__ import annotations

import collections
import copy
import csv
import gc
import hashlib
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

if __package__:
    from . import dataset, losses, model as models, train
else:
    import dataset
    import losses
    import model as models
    import train

IMAGE_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
PAPER_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
CSV_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"


def _checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temp)
    temp.replace(path)


def prepare_data(base_dir):
    """Reuse the previous profiling data, including its legacy /kaggle path."""
    base_dir = Path(base_dir)
    candidates = list(dict.fromkeys([base_dir / "data", Path("/content/data"),
                                    Path("/kaggle/working/data")]))
    data_dir = next((d for d in candidates if (d / "images.zip").is_file()), base_dir / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "images.zip"
    if not archive.exists():
        print("Downloading DeepWeeds images (~490 MB)...", flush=True)
        _download("https://zenodo.org/records/7939060/files/images.zip?download=1", archive)
    if _checksum(archive, "md5") != IMAGE_MD5:
        raise ValueError("images.zip MD5 mismatch; replace the corrupt archive before continuing")
    labels_dir = data_dir / "labels"
    for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
        # Fetch original CSVs. No resplitting, filtering or editing their contents.
        _download(f"{CSV_URL}/{name}.csv", labels_dir / f"{name}.csv")
    with (labels_dir / "train_subset0.csv").open(encoding="utf-8-sig", newline="") as f:
        first = next(csv.DictReader(f))["Filename"]
    locations = list(data_dir.rglob(first))
    if not locations:
        with zipfile.ZipFile(archive) as z:
            root = data_dir.resolve()
            for info in z.infolist():
                if not (root / info.filename).resolve().is_relative_to(root):
                    raise ValueError("Invalid archive path")
            z.extractall(data_dir)
        locations = list(data_dir.rglob(first))
    if len(locations) != 1:
        raise ValueError("Cannot determine a unique images directory")
    images_dir = locations[0].parent
    print("Images:", images_dir, "| CSV:", labels_dir, "| image MD5 verified", flush=True)
    return images_dir, labels_dir


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _show_tensor(tensor, mean=dataset.IMAGENET_MEAN, std=dataset.IMAGENET_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def analyze_data(images_dir, labels_dir, out_dir, seed=0):
    """Test metadata is checked; all displayed/decoded images come from train/val."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.set_seed(seed)
    frames = dataset.load_split(labels_dir)
    checks = dataset.check_split(*frames, images_dir)
    combined = pd.concat(frames, ignore_index=True)
    reference = pd.read_csv(Path(labels_dir) / "labels.csv")
    dataset._validate_frame(reference, "labels.csv")
    joined = combined.set_index("Filename").Label.sort_index()
    original = reference.set_index("Filename").Label.sort_index()
    if set(joined.index) != set(original.index):
        raise ValueError("The split filename union differs from labels.csv")
    # The upstream fold-0 CSVs have one label differing from labels.csv.
    # S1 requires preserving those splits: report the discrepancy, never edit it.
    mismatches = joined[joined != original]
    discrepancy_rows = []
    for filename, label in mismatches.items():
        split = next(s for s, frame in zip(("train", "val", "test"), frames)
                     if filename in set(frame.Filename))
        discrepancy_rows.append({"Filename": filename, "split": split,
                                 "fold0_Label": int(label), "labels_csv_Label": int(original[filename])})
    pd.DataFrame(discrepancy_rows, columns=["Filename", "split", "fold0_Label", "labels_csv_Label"]).to_csv(
        out_dir / "label_discrepancies.csv", index=False)
    counts = combined.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    paper_deltas = counts - np.asarray(PAPER_COUNTS)
    distribution = pd.DataFrame({"class": dataset.CLASS_NAMES,
                                 **{s: checks["per_class"][s] for s in ("train", "val", "test")},
                                 "total": counts, "paper_table1": PAPER_COUNTS,
                                 "delta_vs_paper": paper_deltas})
    distribution.to_csv(out_dir / "class_counts.csv", index=False)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    distribution.set_index("class")[["train", "val", "test"]].plot.bar(ax=ax)
    ax.set(ylabel="Images", title="DeepWeeds fold 0: official class distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(9, 3, figsize=(10, 24))
    for label in range(9):
        selected = frames[0][frames[0].Label == label].sample(3, random_state=seed)
        for axis, (_, row) in zip(axes[label], selected.iterrows()):
            with Image.open(Path(images_dir) / row.Filename) as img:
                axis.imshow(img.convert("RGB"))
            axis.set_title(f"{dataset.CLASS_NAMES[label]}\n{row.Filename}", fontsize=9)
            axis.axis("off")
    fig.suptitle("Three train examples per class (no test images)")
    fig.tight_layout()
    fig.savefig(out_dir / "samples_3_per_class.png", dpi=140)
    plt.close(fig)
    sizes, modes = collections.Counter(), collections.Counter()
    for frame in frames[:2]:
        for filename in frame.Filename:
            with Image.open(Path(images_dir) / filename) as img:
                sizes[str(img.size)] += 1
                modes[img.mode] += 1
    train_df = frames[0]
    balanced = pd.concat([train_df[train_df.Label == c].head(1) for c in range(9)])
    ds = dataset.DeepWeedsDataset(balanced, images_dir, dataset.build_transforms(True))
    batch = [ds[i] for i in range(9)]
    x, y = torch.stack([r[0] for r in batch]), torch.tensor([r[1] for r in batch])
    for kind in ("augmentation", "mixup", "cutmix"):
        if kind == "augmentation":
            shown, targets = x, (y, y, 1.0)
        else:
            shown, targets = losses.mix_batch(x, y, mode=kind)
        a, b, lam = targets
        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        for i, axis in enumerate(axes.flat):
            axis.imshow(_show_tensor(shown[i]))
            title = dataset.CLASS_NAMES[int(a[i])]
            if kind != "augmentation":
                title += f" / {dataset.CLASS_NAMES[int(b[i])]}\nlam={lam:.3f}"
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(f"Train {kind}: denormalized RGB, labels retained")
        fig.tight_layout()
        fig.savefig(out_dir / f"{kind}_examples.png", dpi=140)
        plt.close(fig)
    result = {"split_checks": checks, "global_imbalance_ratio": float(counts.max() / counts.min()),
              "train_imbalance_ratio": max(checks["per_class"]["train"]) / min(checks["per_class"]["train"]),
              "image_header_scope": "train and val only", "image_sizes": dict(sizes),
              "image_modes": dict(modes), "training_conversion": "RGB",
              "reference_filenames_match": True,
              "reference_labels_match": joined.equals(original),
              "label_discrepancies": discrepancy_rows,
              "paper_counts_match": bool(np.array_equal(counts, PAPER_COUNTS)),
              "paper_count_deltas": paper_deltas.tolist(),
              "csv_sha256": {p.name: _checksum(p) for p in Path(labels_dir).glob("*.csv")}}
    train.write_json(out_dir / "eda.json", result)
    print(distribution.to_string(index=False))
    print("Largest/smallest class ratio:", round(result["global_imbalance_ratio"], 2))
    if not result["paper_counts_match"]:
        print("CSV counts differ from Table 1:", dict(zip(dataset.CLASS_NAMES, paper_deltas.tolist())),
              "Keep the original CSVs and report this discrepancy.")
    if discrepancy_rows:
        print("Upstream label discrepancies (fold 0 labels are preserved):", discrepancy_rows)
    print("Train/val image sizes:", dict(sizes), "modes:", dict(modes))
    return result


def pipeline_checks(images_dir, labels_dir, out_dir, *, backbone="resnet50", seed=0,
                    max_steps=200, target_loss=0.05, initial_tolerance=0.75, device="auto"):
    """Overfit a fixed nine-image train batch; discard diagnostic weights afterwards.

    No random augmentation or mixed labels here. Higher diagnostic LR is recorded;
    official experiments still use the unchanged baseline Config learning rates.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.set_seed(seed)
    train_df, _, _ = dataset.load_split(labels_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = models.build_model(backbone, init="finetune").to(device)
    cfg = model.pretrained_cfg
    transform = dataset.build_transforms(False, mean=cfg.get("mean", dataset.IMAGENET_MEAN),
                                          std=cfg.get("std", dataset.IMAGENET_STD),
                                          interpolation=cfg.get("interpolation", "bilinear"))
    samples = pd.concat([train_df[train_df.Label == c].head(3) for c in range(9)])
    initial_loader = dataset.make_loader(samples, images_dir, transform, 27, False,
                                         num_workers=0, seed=seed)
    x_initial, y_initial, _ = next(iter(initial_loader))
    x_initial, y_initial = x_initial.to(device), y_initial.to(device)
    model.eval()
    with torch.no_grad():
        initial_loss = float(torch.nn.functional.cross_entropy(model(x_initial), y_initial))
    expected = math.log(9)
    if abs(initial_loss - expected) > initial_tolerance:
        train.write_json(out_dir / "initial_loss_failure.json",
                         {"initial_loss": initial_loss, "expected": expected,
                          "tolerance": initial_tolerance, "passed": False})
        raise RuntimeError(f"Initial CE {initial_loss:.4f} differs too much from ln(9)={expected:.4f}; inspect the head/data")
    print(f"Initial CE: {initial_loss:.4f}; ln(9): {expected:.4f}", flush=True)
    balanced = pd.concat([train_df[train_df.Label == c].head(1) for c in range(9)])
    loader = dataset.make_loader(balanced, images_dir, transform, 9, False,
                                 num_workers=0, seed=seed)
    x, y, names = next(iter(loader))
    x, y = x.to(device), y.to(device)
    criterion = losses.build_criterion()
    model.train()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not all(m.training for m in bns):
        raise RuntimeError("Unfrozen BatchNorm did not enter train mode")
    probe = copy.deepcopy(model)
    models.freeze_backbone(probe)
    probe.train()
    probe_bns = [m for m in probe.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = [m.running_mean.clone() for m in probe_bns]
    with torch.no_grad():
        probe(x)
    frozen_ok = all(not m.training and torch.equal(old, m.running_mean)
                    for m, old in zip(probe_bns, before)) and probe.get_classifier().training
    if not frozen_ok:
        raise RuntimeError("Frozen backbone BN/head mode check failed")
    del probe
    diagnostic_cfg = train.Config(lr_backbone=1e-3, lr_head=1e-2, weight_decay=0.0)
    optimizer = train.build_optimizer(model, diagnostic_cfg)
    history = []
    succeeded = False
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss in one-batch overfit check")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                eval_loss = float(criterion(logits, y))
                accuracy = float((logits.argmax(1) == y).float().mean())
            history.append({"step": step, "train_loss": float(loss.detach()),
                            "fixed_batch_eval_loss": eval_loss, "fixed_batch_accuracy": accuracy})
            print(f"Fixed batch step {step}: eval CE={eval_loss:.5f}, accuracy={accuracy:.3f}", flush=True)
            if eval_loss <= target_loss and accuracy == 1.0:
                succeeded = True
                break
    pd.DataFrame(history).to_csv(out_dir / "overfit_history.csv", index=False)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in history], [r["train_loss"] for r in history], label="train CE")
    ax.plot([r["step"] for r in history], [r["fixed_batch_eval_loss"] for r in history], label="fixed batch eval CE")
    ax.set(xlabel="Optimizer steps", ylabel="CE loss", title="Diagnostic: overfit 9 fixed train images", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overfit_curve.png", dpi=160)
    plt.close(fig)
    result = {"passed": succeeded, "seed": seed, "backbone": backbone,
              "device": str(device), "initial_loss": initial_loss, "expected_initial_ce": expected,
              "initial_tolerance": initial_tolerance, "overfit_target_loss": target_loss,
              "steps_used": step, "final_fixed_batch_loss": history[-1]["fixed_batch_eval_loss"],
              "final_fixed_batch_accuracy": history[-1]["fixed_batch_accuracy"],
              "frozen_bn_mode_passed": frozen_ok, "diagnostic_lr_backbone": 1e-3,
              "diagnostic_lr_head": 1e-2, "batch_filenames": list(names),
              "diagnostic_weights_reused_for_experiments": False}
    train.write_json(out_dir / "pipeline_checks.json", result)
    del model, optimizer, x, y, x_initial, y_initial
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not succeeded:
        raise RuntimeError("Fixed batch did not overfit; investigate before running official experiments")
    return result


def write_report(out_dir, eda, checks, baseline):
    out_dir = Path(out_dir)
    counts = eda["split_checks"]["n"]
    text = f"""# Bước 0 — Bằng chứng từ lần chạy thật

## Dữ liệu
Fold 0 nguyên bản. Train: {counts['train']}; val: {counts['val']}; test: {counts['test']}.
Giao từng cặp: {eda['split_checks']['overlap']}; hợp: {eda['split_checks']['union']}; thiếu file: 0.
Khớp nhãn trong labels.csv: {eda['reference_labels_match']}; khớp Table 1: {eda['paper_counts_match']}.
Hợp tên ảnh khớp labels.csv: {eda['reference_filenames_match']}.
Số nhãn fold 0 khác labels.csv: {len(eda['label_discrepancies'])}; chi tiết trong label_discrepancies.csv.
Chênh số ảnh từng lớp so với Table 1 (thứ tự Label 0..8): {eda['paper_count_deltas']}.
Giữ nguyên CSV tác giả; không sửa/lọc ảnh để ép số đếm khớp bảng tham khảo.
Tỉ lệ lớp lớn/nhỏ toàn dataset: {eda['global_imbalance_ratio']:.3f}.
Chỉ xem ảnh train và thông tin ảnh train/val. Không đánh giá test.
Biểu đồ: class_distribution.png; ảnh mẫu: samples_3_per_class.png (3 ảnh/lớp).
Ảnh augmentation/Mixup/CutMix có nhãn nằm trong các file *_examples.png.

## Kiểm tra pipeline
Seed: {checks['seed']}; backbone: {checks['backbone']}.
CE ban đầu trên 27 ảnh train cân bằng: {checks['initial_loss']:.6f}; ln(9): {checks['expected_initial_ce']:.6f}.
Độ lệch cho phép cho head ngẫu nhiên: {checks['initial_tolerance']} (đây là kiểm tra sơ bộ).
Overfit 9 ảnh cố định: eval CE {checks['final_fixed_batch_loss']:.6f}, accuracy {checks['final_fixed_batch_accuracy']:.3f}, {checks['steps_used']} bước.
LR chẩn đoán: backbone 1e-3, head 1e-2, không augmentation/weight decay; trọng số này bị bỏ đi.
BN khi đóng băng giữ eval, head giữ train: {checks['frozen_bn_mode_passed']}.
Unit tests riêng kiểm tra focal gamma=0, CutMix/nhãn/diện tích và hợp đồng ghi dự đoán.

## Lần chạy chung run(Config(...))
{baseline['exp_id']}, seed {baseline['seed']}, backbone {baseline['backbone']}.
Checkpoint tốt nhất: epoch {baseline['best_epoch']}, chọn hoàn toàn bằng macro-F1 val.
Macro-F1 val: {baseline['macro_f1_val']:.6f}; top-1 val: {baseline['top1_val']:.6f}.
Tham số: {baseline['params_m']:.3f} triệu; GMAC: {baseline['gmacs']:.4f} (fvcore; xem config.json để biết ops chưa đếm).
history.csv, config.json, best.pt/last.pt và val_logits.npy nằm trong runs/{baseline['exp_id']}/seed{baseline['seed']}/.
Dự đoán val: {baseline['val_predictions']}; đường cong: {baseline['curve']}.
Test đã chạy: {baseline['test_evaluated']}.

## Nhận xét cần bổ sung bằng mắt
- Điền nhận xét từ ảnh mẫu về Chinee Apple / Snake Weed và lớp Negatives.
- Điền nhận xét về crop và CutMix có giữ được đối tượng cỏ mục tiêu không.
- Chưa có kết luận so sánh backbone; các vòng B/T/I/F tiếp theo mới trả lời điều đó.
"""
    (out_dir / "step0_report.md").write_text(text, encoding="utf-8")


def export_artifacts(project_dir, out_path):
    """Bundle Step 0 code/results without images or checkpoint files."""
    project_dir, out_path = Path(project_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in ("code", "tests", "step0", "runs", "curves", "predictions"):
            for path in sorted((project_dir / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".md", ".json", ".csv", ".png", ".npy", ".ipynb"):
                    bundle.write(path, path.relative_to(project_dir))
        bundle.write(project_dir / "eval.py", "eval.py")
    return out_path
