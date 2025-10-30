from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from trainer.trainer import Trainer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 detector.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/yolo11.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.project is not None:
        cfg.project = args.project
    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(cfg, device=device, work_dir=cfg.project)
    trainer.train()


if __name__ == "__main__":
    main()
