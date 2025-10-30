import pathlib
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def letterbox(img, new_shape, color=(114, 114, 114)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    if new_unpad != (shape[1], shape[0]):
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2

    if left or right or top or bottom:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = (new_unpad[0] / shape[1], new_unpad[1] / shape[0])
    pad = (left, top)
    return img, ratio, pad


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, size=3) * np.array([hgain, sgain, vgain]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

    x = np.arange(256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)

class Datasetloader(Dataset):
    def __init__(
        self,
        root,
        split,
        img_size=640,
        augment=True,
        mosaic_prob=0.5,
        mixup_prob=0.15,
    ):
        self.root = pathlib.Path(root)
        self.img_dir = self.root / f"images/{split}"
        self.label_dir = self.root / f"labels/{split}"
        self.files = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.mixup_prob = mixup_prob if augment else 0.0

    def __len__(self):
        return len(self.files)

    def _load_image_and_targets(self, idx):
        stem = self.files[idx]
        img_path = self.img_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_path = self.label_dir / f"{stem}.txt"
        targets = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.split())
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    targets.append([cls, x1, y1, x2, y2])

        targets = np.array(targets, dtype=np.float32) if targets else np.zeros((0, 5), dtype=np.float32)
        return img, targets

    def _letterbox_image(self, img, targets):
        img, ratio, pad = letterbox(img, self.img_size)

        if targets.size:
            targets = targets.copy()
            targets[:, [1, 3]] = targets[:, [1, 3]] * ratio[0] + pad[0]
            targets[:, [2, 4]] = targets[:, [2, 4]] * ratio[1] + pad[1]
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        return img, targets

    def _clip_targets(self, targets, size):
        if not targets.size:
            return targets

        targets[:, [1, 3]] = targets[:, [1, 3]].clip(0, size - 1)
        targets[:, [2, 4]] = targets[:, [2, 4]].clip(0, size - 1)

        widths = targets[:, 3] - targets[:, 1]
        heights = targets[:, 4] - targets[:, 2]
        keep = (widths > 2) & (heights > 2)
        return targets[keep]

    def _load_mosaic(self, idx):
        s = self.img_size
        yc = random.randint(s // 2, s + s // 2)
        xc = random.randint(s // 2, s + s // 2)
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_targets = []

        indices = [idx] + random.choices(range(len(self.files)), k=3)
        for i, index in enumerate(indices):
            img, targets = self._load_image_and_targets(index)
            if targets.size:
                targets = targets.copy()

            h, w = img.shape[:2]
            scale = s / max(h, w)
            if scale != 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                if targets.size:
                    targets[:, [1, 3]] *= scale
                    targets[:, [2, 4]] *= scale

            h, w = img.shape[:2]

            if i == 0:  # top-left
                x1a = max(xc - w, 0)
                y1a = max(yc - h, 0)
                x2a = xc
                y2a = yc
                x1b = w - (x2a - x1a)
                y1b = h - (y2a - y1a)
                x2b = w
                y2b = h
            elif i == 1:  # top-right
                x1a = xc
                y1a = max(yc - h, 0)
                x2a = min(xc + w, s * 2)
                y2a = yc
                x1b = 0
                y1b = h - (y2a - y1a)
                x2b = x2a - x1a
                y2b = h
            elif i == 2:  # bottom-left
                x1a = max(xc - w, 0)
                y1a = yc
                x2a = xc
                y2a = min(yc + h, s * 2)
                x1b = w - (x2a - x1a)
                y1b = 0
                x2b = w
                y2b = y2a - y1a
            else:  # bottom-right
                x1a = xc
                y1a = yc
                x2a = min(xc + w, s * 2)
                y2a = min(yc + h, s * 2)
                x1b = 0
                y1b = 0
                x2b = x2a - x1a
                y2b = y2a - y1a

            if x2a <= x1a or y2a <= y1a:
                continue

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            if targets.size:
                boxes = targets.copy()
                boxes[:, [1, 3]] += x1a - x1b
                boxes[:, [2, 4]] += y1a - y1b
                mosaic_targets.append(boxes)

        if mosaic_targets:
            mosaic_targets = np.concatenate(mosaic_targets, axis=0)
        else:
            mosaic_targets = np.zeros((0, 5), dtype=np.float32)

        x_offset = random.randint(0, s)
        y_offset = random.randint(0, s)
        img = mosaic_img[y_offset:y_offset + s, x_offset:x_offset + s]

        if mosaic_targets.size:
            mosaic_targets[:, [1, 3]] -= x_offset
            mosaic_targets[:, [2, 4]] -= y_offset
            mosaic_targets = self._clip_targets(mosaic_targets, s)
        else:
            mosaic_targets = np.zeros((0, 5), dtype=np.float32)

        return img, mosaic_targets

    def __getitem__(self, idx):
        if self.augment and random.random() < self.mosaic_prob:
            img, targets = self._load_mosaic(idx)
            if self.mixup_prob > 0 and random.random() < self.mixup_prob:
                mix_idx = random.randint(0, len(self.files) - 1)
                mix_img, mix_targets = self._load_mosaic(mix_idx)
                ratio = np.random.beta(8.0, 8.0)
                img = (img.astype(np.float32) * ratio + mix_img.astype(np.float32) * (1.0 - ratio)).astype(np.uint8)
                if mix_targets.size:
                    targets = np.concatenate((targets, mix_targets), axis=0) if targets.size else mix_targets
        else:
            img, targets = self._load_image_and_targets(idx)
            img, targets = self._letterbox_image(img, targets)

        if self.augment:
            augment_hsv(img)
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                if targets.size:
                    width = img.shape[1]
                    x1 = targets[:, 1].copy()
                    x2 = targets[:, 3].copy()
                    targets[:, 1] = width - x2
                    targets[:, 3] = width - x1
            targets = self._clip_targets(targets, img.shape[1])
        else:
            targets = self._clip_targets(targets, img.shape[1])

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div(255.0)
        targets_tensor = torch.from_numpy(targets) if targets.size else torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, targets_tensor

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets
