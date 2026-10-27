#!/usr/bin/env python3
"""
Train LabRA-UNet.

Defaults: CIELAB input resized to 256 x 256, Focal-Tversky loss, AdamW (learning rate
0.001, weight decay 0.01) with cosine annealing, batch size 16, at most 200 epochs.
Training images whose mask contains LC are oversampled x4. After every epoch the model
is evaluated on the validation set; the checkpoint with the best validation Dice
(mean of BP and LC) is kept, and training stops when the validation Dice has not
improved for 30 epochs.

Usage: python train.py --data Bacteria-35-split
Writes runs/train/expN/: best_model.pth, log.csv, training_curves.png, train_info.json
"""
import argparse, json, os, random, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import (Compose, ElasticTransform, GridDistortion, HorizontalFlip, Normalize,
                            RandomRotate90, Resize, VerticalFlip)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import LeafDataset
from losses import FocalTverskyLoss
from models import LabRAUNet


def transforms(imgsz, train):
    aug = [HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5),
           GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
           ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3)] if train else []
    return Compose([Resize(imgsz, imgsz), *aug, Normalize((0.5,) * 3, (0.5,) * 3), ToTensorV2()])


@torch.no_grad()
def evaluate(model, loader, dev, amp, amp_dtype):
    """Pixel-level Dice, IoU, precision and recall over the whole set, per lesion class."""
    model.eval()
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    for x, y in loader:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            pred = model(x.to(dev, non_blocking=True)).argmax(1).cpu()
        for c in range(3):
            p, t = pred == c, y == c
            tp[c] += (p & t).sum().item(); fp[c] += (p & ~t).sum().item(); fn[c] += (~p & t).sum().item()
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec, rec = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return {"val_dice": dice[1:].mean(), "val_dice_BP": dice[1], "val_dice_LC": dice[2],
            "val_iou_BP": iou[1], "val_iou_LC": iou[2],
            "val_precision": prec[1:].mean(), "val_recall": rec[1:].mean()}


def plot_curves(log, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log["epoch"], log["train_loss"], color="#C0392B")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss (Focal-Tversky)")
    for col, lab, c in (("val_dice", "Dice (mean)", "black"), ("val_dice_BP", "Dice BP", "#E41A1C"),
                        ("val_dice_LC", "Dice LC", "#1F45FC")):
        ax[1].plot(log["epoch"], log[col], label=lab, color=c, lw=1.6 if col == "val_dice" else 1)
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation Dice"); ax[1].legend(frameon=False)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ and valid/ (images, masks)")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--oversample-lc", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3,
                    help="data-loading processes for training (each loads PyTorch; keep low on Windows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="train in float32 instead of mixed precision")
    ap.add_argument("--project", default="runs/train")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.no_amp
    # bfloat16 has the range of float32, so activations cannot overflow as they can in float16
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    os.makedirs(a.project, exist_ok=True)
    n = 1 + max([int(d[3:]) for d in os.listdir(a.project) if d.startswith("exp") and d[3:].isdigit()] or [0])
    out = os.path.join(a.project, f"exp{n}"); os.makedirs(out)

    def loader(split, train):
        ds = LeafDataset(os.path.join(a.data, split, "images"), os.path.join(a.data, split, "masks"),
                         transforms(a.imgsz, train), a.oversample_lc if train else 0)
        workers = a.workers if train else 0          # the validation set is small: load it in the main process
        return ds, DataLoader(ds, a.batch, shuffle=train, drop_last=train, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    tr_ds, tr = loader("train", True)
    va_ds, va = loader("valid", False)

    model = LabRAUNet(n_cls=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    crit = FocalTverskyLoss()
    print(f"{out}: {len(tr_ds)} training samples, {len(va_ds)} validation images, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)

    best, best_epoch, bad, log, t0 = -1.0, 0, 0, [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item()
        lr = opt.param_groups[0]["lr"]; sched.step()
        row = {"epoch": epoch, "lr": lr, "train_loss": total / len(tr), **evaluate(model, va, dev, amp, amp_dtype)}
        log.append(row); pd.DataFrame(log).to_csv(os.path.join(out, "log.csv"), index=False)
        if row["val_dice"] > best:
            best, best_epoch, bad = row["val_dice"], epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best_model.pth"))
        else:
            bad += 1
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val Dice {row['val_dice']:.4f} "
              f"(BP {row['val_dice_BP']:.4f}, LC {row['val_dice_LC']:.4f})  {'*' if bad == 0 else ''}", flush=True)
        if not np.isfinite(row["train_loss"]):
            print("stopping: the training loss is not finite", flush=True); break
        if bad >= a.patience:
            print(f"early stopping: no improvement for {a.patience} epochs", flush=True); break

    plot_curves(pd.DataFrame(log), os.path.join(out, "training_curves.png"))
    info = {**vars(a), "train_samples": len(tr_ds), "valid_images": len(va_ds), "epochs_run": len(log),
            "best_epoch": best_epoch, "best_val_dice": best, "train_hours": (time.time() - t0) / 3600,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "torch": torch.__version__, "mixed_precision": str(amp_dtype).replace("torch.", "") if amp else False}
    json.dump(info, open(os.path.join(out, "train_info.json"), "w"), indent=2)
    print(f"best validation Dice {best:.4f} at epoch {best_epoch}; {info['train_hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
