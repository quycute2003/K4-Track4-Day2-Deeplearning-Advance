"""One training entry point for DeepWeeds B/T/F experiments.

Windows: py -X utf8 code/train.py --set exp_id=B01 backbone=resnet50 epochs=10 batch_size=32
Evaluation and prediction serialization come from the unmodified root eval.py.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import importlib.metadata
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pandas as pd
import torch

if __package__:
    from . import dataset, losses, model as models
else:
    import dataset
    import losses
    import model as models

_eval_path = Path(__file__).resolve().parent.parent / "eval.py"
_eval_spec = importlib.util.spec_from_file_location("_deepweeds_official_eval", _eval_path)
ev = importlib.util.module_from_spec(_eval_spec)
sys.modules[_eval_spec.name] = ev
_eval_spec.loader.exec_module(ev)
compute_metrics, save_predictions = ev.compute_metrics, ev.save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    curves_dir: str = "curves"
    device: str = "auto"
    resume: bool = False
    grad_clip_norm: float | None = None


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg):
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head,
                                                cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if steps_per_epoch <= 0 or cfg.epochs <= 0 or cfg.warmup_epochs < 0:
        raise ValueError("Invalid scheduler duration")
    total = cfg.epochs * steps_per_epoch
    warmup = min(total, round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Average parameters; copy buffers, including BN stats, from the live model."""
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        live = dict(model.named_parameters())
        for name, p in self.module.named_parameters():
            p.lerp_(live[name].detach().to(p), 1 - self.decay)
        live_buffers = dict(model.named_buffers())
        for name, buffer in self.module.named_buffers():
            buffer.copy_(live_buffers[name].detach())

    def copy_to(self, model):
        model.load_state_dict(self.module.state_dict())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg,
                    device, ema=None):
    model.train()  # Frozen models override train() to keep their features in eval.
    loss_sum, n = 0.0, 0
    amp = cfg.amp and device.type == "cuda"
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y) if targets is None else losses.mixed_loss(criterion, logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        previous_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step on overflow; don't advance LR/EMA then.
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        loss_sum += float(loss.detach()) * len(y)
        n += len(y)
    if n == 0:
        raise ValueError("Empty training loader")
    return {"train_loss": loss_sum / n, "lr": optimizer.param_groups[0]["lr"],
            "train_images_seen": n}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Return filenames, labels, float32 logits and mean batch loss; preserve order."""
    model.eval()
    filenames, labels, logits_all = [], [], []
    loss_sum, n = 0.0, 0
    for x, y, names in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        loss = criterion(logits, y)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation outputs")
        filenames.extend(names)
        labels.append(y.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        loss_sum += float(loss) * len(y)
        n += len(y)
    if n == 0:
        raise ValueError("Empty evaluation loader")
    return filenames, np.concatenate(labels), np.concatenate(logits_all), loss_sum / n


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train")
    axes[0].plot(frame.epoch, frame.val_loss, label="val")
    axes[0].set(ylabel="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.val_macro_f1, label="macro-F1 val")
    axes[1].plot(frame.epoch, frame.val_top1, label="top-1 val")
    axes[1].set(ylabel="Score", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    axes[2].plot(frame.epoch, frame.lr)
    axes[2].set(ylabel="LR (first optimizer group)", xlabel="Epoch")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _save_checkpoint(path, value):
    temp = Path(path).with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _rng_state(loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": loader.generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    loader.generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def run(cfg):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg.exp_id):
        raise ValueError("exp_id must be a simple filename-safe identifier")
    if cfg.epochs <= 0 or cfg.fold not in range(5):
        raise ValueError("epochs must be positive and fold 0..4")
    if cfg.save_test_predictions and pred_path(cfg, "test").exists():
        raise FileExistsError("Test predictions already exist; do not rerun test to select results")
    folder = run_dir(cfg)
    latest = folder / "last.pt"
    if cfg.resume:
        if not latest.exists():
            raise FileNotFoundError("resume=True needs an existing last.pt")
        previous = json.loads((folder / "config.json").read_text(encoding="utf-8"))["config"]
        current = asdict(cfg)
        if any(previous[k] != current[k] for k in current if k != "resume"):
            raise ValueError("Resume configuration differs from the original run")
    elif latest.exists() or (folder / "history.csv").exists():
        raise FileExistsError("Run already exists: use resume=True or a new exp_id")
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_info = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    write_json(folder / "split_checks.json", split_info)
    model = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate,
                               img_size=cfg.img_size).to(device)
    pretrained_cfg = dict(model.pretrained_cfg)
    transform_kw = {"mean": pretrained_cfg.get("mean", dataset.IMAGENET_MEAN),
                    "std": pretrained_cfg.get("std", dataset.IMAGENET_STD),
                    "interpolation": pretrained_cfg.get("interpolation", "bilinear")}
    train_tf = dataset.build_transforms(True, cfg.img_size, cfg.aug, **transform_kw)
    eval_tf = dataset.build_transforms(False, cfg.img_size, **transform_kw)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size,
                                       True, cfg.sampler, cfg.num_workers, seed=cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size,
                                     False, num_workers=cfg.num_workers, seed=cfg.seed)
    weights = None
    if cfg.loss == "ce_weighted" or (cfg.loss == "focal" and cfg.class_weight_beta is not None):
        counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
        weights = losses.class_weights(counts, cfg.class_weight_beta or 0.0)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weights, alpha=weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    import timm
    import torchvision
    params_m, gmacs = models.count_params(model), models.count_gmacs(model, cfg.img_size)
    record = {"config": asdict(cfg), "pretrained_cfg": pretrained_cfg,
              "versions": {"python": sys.version, "torch": str(torch.__version__),
                           "torchvision": torchvision.__version__, "timm": timm.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__,
                           "matplotlib": importlib.metadata.version("matplotlib"),
                           "Pillow": importlib.metadata.version("Pillow"),
                           "fvcore": importlib.metadata.version("fvcore")},
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "params_m": params_m, "gmacs": gmacs, "gmac_report": model.gmac_report,
              "selection": "val macro-F1, earliest epoch on ties",
              "evaluation_dtype": "FP32", "seed_policy": "Python/NumPy/torch/worker seeded; cudnn deterministic",
              "ema_buffers": "copy live model buffers" if ema else None,
              "test_evaluated": False}
    write_json(folder / "config.json", record)
    history, best_f1, best_epoch, start_epoch = [], -1.0, 0, 0
    if cfg.resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.module.load_state_dict(state["ema"])
        history, best_f1, best_epoch = state["history"], state["best_f1"], state["best_epoch"]
        start_epoch = state["epoch"]
        _restore_rng(state["rng"], train_loader)
    curve = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png"
    for epoch in range(start_epoch, cfg.epochs):
        _sync(device)
        start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                      scaler, cfg, device, ema)
        _sync(device)
        train_seconds = time.perf_counter() - start
        selected_model = ema.module if ema else model
        _, y_true, logits, val_loss = evaluate(selected_model, val_loader, criterion, device)
        probs = probabilities(logits)
        metric = compute_metrics(y_true, probs.argmax(1), probs)
        row = {"epoch": epoch + 1, **train_stats, "val_loss": val_loss,
               "val_macro_f1": metric["macro_f1"], "val_top1": metric["top1"],
               "val_balanced_acc": metric["balanced_acc"], "val_ece": metric["ece"],
               "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - start}
        history.append(row)
        if metric["macro_f1"] > best_f1:
            best_f1, best_epoch = metric["macro_f1"], epoch + 1
            _save_checkpoint(folder / "best.pt", {"model": selected_model.state_dict(),
                             "epoch": best_epoch, "macro_f1": best_f1, "ema": ema is not None})
        _save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "ema": ema.module.state_dict() if ema else None,
                         "epoch": epoch + 1, "history": history, "best_f1": best_f1,
                         "best_epoch": best_epoch, "rng": _rng_state(train_loader)})
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        plot_curves(history, curve, f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
        print(f"Epoch {epoch + 1}/{cfg.epochs}: loss={row['train_loss']:.4f}, "
              f"macro-F1 val={best_f1:.4f} (best), current={metric['macro_f1']:.4f}", flush=True)
    state = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    names, labels, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
    np.save(folder / "val_logits.npy", val_logits)
    np.save(folder / "val_labels.npy", labels)
    write_json(folder / "val_filenames.json", names)
    val_probs = probabilities(val_logits)
    save_predictions(pred_path(cfg, "val"), names, labels, val_probs)
    val_metrics = compute_metrics(labels, val_probs.argmax(1), val_probs)
    result = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
              "best_epoch": best_epoch, "macro_f1_val": val_metrics["macro_f1"],
              "top1_val": val_metrics["top1"], "ece_val": val_metrics["ece"],
              "val_loss": val_loss, "params_m": params_m, "gmacs": gmacs,
              "mean_train_seconds": float(np.mean([h["train_seconds"] for h in history])),
              "val_predictions": str(pred_path(cfg, "val")), "curve": str(curve),
              "test_evaluated": False}
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, eval_tf, cfg.batch_size,
                                          False, num_workers=cfg.num_workers, seed=cfg.seed)
        names, labels, logits, _ = evaluate(model, test_loader, criterion, device)
        np.save(folder / "test_logits.npy", logits)
        save_predictions(pred_path(cfg, "test"), names, labels, probabilities(logits))
        result["test_evaluated"] = record["test_evaluated"] = True
        write_json(folder / "config.json", record)
    write_json(folder / "summary.json", result)
    return result


def parse_overrides(pairs):
    hints = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair}")
        key, value = pair.split("=", 1)
        if key not in hints:
            raise ValueError(f"Unknown Config field: {key}")
        hint = hints[key]
        choices = get_args(hint) or (hint,)
        if value.lower() in ("none", "null") and type(None) in choices:
            parsed = None
        elif bool in choices:
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key}: use true/false or 1/0")
            parsed = value.lower() in ("true", "1")
        elif int in choices:
            parsed = int(value)
        elif float in choices:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"{key}: must be finite")
        else:
            parsed = value
        result[key] = parsed
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    try:
        cfg = Config(**parse_overrides(args.set))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(run(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
