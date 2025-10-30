from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import importlib

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.backbone import Yolo11BackBone
from model.neck import Yolo11Neck
from model.modules.detect import AFDetect
from utils.datasets import Datasetloader, collate_fn
from utils.loss import ComputeTalLoss


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: os.PathLike[str] | str) -> SimpleNamespace:
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to parse training configuration files.") from exc
    with open(path, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)
    return _to_namespace(cfg_dict)


class Yolo11(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.backbone = Yolo11BackBone(cfg)
        self.neck = Yolo11Neck(cfg)
        nc = getattr(cfg.Dataset, "nc", 1)

        head_channels = [
            self.neck.channels["out_p3"],
            self.neck.channels["out_p4"],
            self.neck.channels["out_p5"],
        ]
        self.head = AFDetect(nc=nc, nl=len(head_channels), ch=head_channels)
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        features = self.backbone(x)
        neck_out = self.neck(features)
        return self.head(neck_out)


class Trainer:
    def __init__(
        self,
        cfg: SimpleNamespace,
        device: Optional[torch.device] = None,
        work_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.work_dir = Path(work_dir or getattr(cfg, "project", "runs")).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = getattr(cfg, "epochs", 100)
        self.start_epoch = 0
        self.global_step = 0
        hyp_cfg = getattr(cfg, "hyp", SimpleNamespace())
        default_interval = getattr(hyp_cfg, "log_interval", 50)
        self.log_interval = max(1, int(getattr(cfg, "log_interval", default_interval)))

        self.model = Yolo11(cfg).to(self.device)
        self.criterion = ComputeTalLoss(self.model, cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.best_val = float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr0 = getattr(self.cfg.hyp, "lr0", 0.01)
        weight_decay = getattr(self.cfg.hyp, "weight_decay", 5e-4)
        if getattr(self.cfg, "adam", False):
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr0,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
        momentum = getattr(self.cfg.hyp, "momentum", 0.937)
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        lr0 = getattr(self.cfg.hyp, "lr0", 0.01)
        lrf = getattr(self.cfg.hyp, "lrf", 0.2)
        eta_min = lr0 * lrf
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=eta_min)

    def _build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        ds_cfg = self.cfg.Dataset
        root = Path(getattr(ds_cfg, "data_name", ".")).expanduser()
        img_size = getattr(ds_cfg, "img_size", 640)
        batch_size = getattr(ds_cfg, "batch_size", 16)
        workers = getattr(ds_cfg, "workers", min(os.cpu_count() or 4, 8))
        mosaic_prob = getattr(ds_cfg, "mosaic_prob", 0.5)
        mixup_prob = getattr(ds_cfg, "mixup_prob", getattr(self.cfg.hyp, "mixup", 0.15))

        train_dataset = Datasetloader(
            root=root,
            split="train",
            img_size=img_size,
            augment=True,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

        val_loader: Optional[DataLoader] = None
        if getattr(ds_cfg, "val", None) is not None:
            val_dataset = Datasetloader(
                root=root,
                split="val",
                img_size=img_size,
                augment=False,
                mosaic_prob=0.0,
                mixup_prob=0.0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=workers,
                collate_fn=collate_fn,
            )

        return train_loader, val_loader

    def train(self) -> None:
        print(
            f"Starting training for {self.epochs} epochs on device {self.device}. "
            f"Batches per epoch: {len(self.train_loader)}. Log interval: {self.log_interval}."
        )
        if self.val_loader is not None:
            print(f"Validation enabled with {len(self.val_loader)} batches per epoch.")

        for epoch in range(self.start_epoch, self.epochs):
            train_log = self._train_one_epoch(epoch)
            self.scheduler.step()

            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate()
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save_checkpoint(epoch, best=True)

            self._save_checkpoint(epoch, best=False)
            lr = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch + 1}/{self.epochs} | lr: {lr:.5f} | train_loss: {train_log['loss']:.4f}"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f}"
            print(msg)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        cls_loss = 0.0
        box_loss = 0.0
        dfl_loss = 0.0
        num_batches = len(self.train_loader)

        for step, (images, target_list) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = [t.to(self.device) for t in target_list]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                preds = self.model(images)
                loss, loss_items = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.cfg.hyp, "max_grad_norm", 10.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            cls_loss += float(loss_items["loss_cls"])
            box_loss += float(loss_items["loss_iou"])
            dfl_loss += float(loss_items["loss_dfl"])
            self.global_step += 1
            if (step == 0) or ((step + 1) % self.log_interval == 0) or (step + 1 == num_batches):
                avg_loss = epoch_loss / (step + 1)
                avg_cls = cls_loss / (step + 1)
                avg_iou = box_loss / (step + 1)
                avg_dfl = dfl_loss / (step + 1)
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] Step [{step + 1}/{num_batches}] "
                    f"loss: {avg_loss:.4f} | cls: {avg_cls:.4f} | iou: {avg_iou:.4f} | dfl: {avg_dfl:.4f}"
                )

        return {
            "loss": epoch_loss / num_batches,
            "loss_cls": cls_loss / num_batches,
            "loss_iou": box_loss / num_batches,
            "loss_dfl": dfl_loss / num_batches,
        }

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, target_list in self.val_loader:  # type: ignore[arg-type]
                images = images.to(self.device, non_blocking=True)
                targets = [t.to(self.device) for t in target_list]
                preds = self.model(images)
                loss, _ = self.criterion(preds, targets)
                total_loss += loss.item()
        return total_loss / max(len(self.val_loader), 1)  # type: ignore[arg-type]

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_val": self.best_val,
            "global_step": self.global_step,
            "config": self.cfg,
        }
        filename = "best.pt" if best else f"epoch_{epoch + 1}.pt"
        torch.save(state, self.work_dir / filename)


__all__ = ["Trainer", "Yolo11", "load_config"]
