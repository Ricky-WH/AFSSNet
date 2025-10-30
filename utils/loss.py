from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.modules import DFL
from utils.tal import dist2bbox, make_anchors


def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return box1.new_tensor(0.0)
    x1 = torch.maximum(box1[:, 0], box2[:, 0])
    y1 = torch.maximum(box1[:, 1], box2[:, 1])
    x2 = torch.minimum(box1[:, 2], box2[:, 2])
    y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp_(0) * (y2 - y1).clamp_(0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp_(0) * (box1[:, 3] - box1[:, 1]).clamp_(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_(0) * (box2[:, 3] - box2[:, 1]).clamp_(0)
    union = area1 + area2 - inter + eps
    return inter / union


def iou_calculator(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    box1 = box1.unsqueeze(2)
    box2 = box2.unsqueeze(1)
    lt = torch.maximum(box1[..., :2], box2[..., :2])
    rb = torch.minimum(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp_(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (box1[..., 2] - box1[..., 0]).clamp_(min=0) * (box1[..., 3] - box1[..., 1]).clamp_(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp_(min=0) * (box2[..., 3] - box2[..., 1]).clamp_(min=0)
    union = area1 + area2 - overlap + eps
    return overlap / union


def select_candidates_in_gts(anchor_points: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if gt_bboxes.numel() == 0:
        return gt_bboxes.new_zeros(gt_bboxes.shape[0], 0, anchor_points.shape[0])
    num_anchors = anchor_points.shape[0]
    bs, max_boxes, _ = gt_bboxes.shape
    points = anchor_points.view(1, 1, num_anchors, 2)
    lt = points - gt_bboxes[:, :, None, 0:2]
    rb = gt_bboxes[:, :, None, 2:4] - points
    deltas = torch.cat([lt, rb], dim=-1)
    return (deltas.min(dim=-1).values > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, max_boxes: int):
    fg_mask = mask_pos.sum(dim=-2)
    if fg_mask.max() > 1:
        mask_multi = (fg_mask.unsqueeze(1) > 1).expand_as(mask_pos)
        max_overlaps_idx = overlaps.argmax(dim=1)
        is_max = F.one_hot(max_overlaps_idx, max_boxes).permute(0, 2, 1).to(mask_pos.dtype)
        mask_pos = torch.where(mask_multi, is_max, mask_pos)
        fg_mask = mask_pos.sum(dim=-2)
    target_gt_idx = mask_pos.argmax(dim=-2)
    return target_gt_idx, fg_mask, mask_pos


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    lt = anchor_points - bbox[..., :2]
    rb = bbox[..., 2:] - anchor_points
    dist = torch.cat([lt, rb], dim=-1)
    return dist.clamp_(0, reg_max - 1e-3)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, top_k: int, num_classes: int, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        bs, max_boxes, _ = gt_bboxes.shape
        if max_boxes == 0 or mask_gt.sum() == 0:
            device = pd_scores.device
            anchors = pd_scores.shape[1]
            return (
                torch.full((bs, anchors), self.bg_idx, device=device, dtype=torch.long),
                torch.zeros((bs, anchors, 4), device=device, dtype=pd_bboxes.dtype),
                torch.zeros((bs, anchors, self.num_classes), device=device, dtype=pd_scores.dtype),
                torch.zeros((bs, anchors), device=device, dtype=torch.bool),
            )

        mask_gt = mask_gt.to(pd_scores.dtype)
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric = align_metric * mask_pos
        pos_align = align_metric.max(dim=-1, keepdim=True).values
        pos_overlaps = (overlaps * mask_pos).max(dim=-1, keepdim=True).values
        norm_align = (align_metric * pos_overlaps / (pos_align + self.eps)).max(dim=-2).values.unsqueeze(-1)
        target_scores = target_scores * norm_align

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        mask_topk = self.select_topk_candidates(align_metric)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ):
        bs, max_boxes, _ = gt_bboxes.shape
        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.long().clamp_(0, self.num_classes - 1)
        batch_idx = torch.arange(bs, device=gt_labels.device).view(bs, 1).expand(-1, max_boxes)
        bbox_scores = pd_scores[batch_idx, gt_labels.squeeze(-1)]
        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics: torch.Tensor):
        bs, max_boxes, num_anchors = metrics.shape
        k = min(self.topk, num_anchors)
        if k <= 0:
            return metrics.new_zeros(metrics.shape)
        topk_idxs = metrics.topk(k, dim=-1).indices
        mask = metrics.new_zeros(metrics.shape, dtype=torch.bool)
        mask.scatter_(-1, topk_idxs, True)
        return mask.to(metrics.dtype)

    def get_targets(self, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, target_gt_idx: torch.Tensor, fg_mask: torch.Tensor):
        bs, anchors = target_gt_idx.shape
        max_boxes = gt_bboxes.shape[1]
        batch_offset = torch.arange(bs, device=gt_labels.device)[:, None] * max_boxes
        flat_idx = (target_gt_idx + batch_offset).view(-1)
        flat_labels = gt_labels.view(-1, 1)
        target_labels = flat_labels[flat_idx].view(bs, anchors, 1).squeeze(-1)
        flat_bboxes = gt_bboxes.view(-1, 4)
        target_bboxes = flat_bboxes[flat_idx].view(bs, anchors, 4)
        target_labels = torch.where(fg_mask, target_labels, target_labels.new_full(target_labels.shape, self.bg_idx))
        target_scores = F.one_hot(target_labels.clamp_(0, self.num_classes - 1), self.num_classes).to(gt_bboxes.dtype)
        target_scores = target_scores * fg_mask.unsqueeze(-1).float()
        return target_labels.long(), target_bboxes, target_scores


class ComputeTalLoss:
    def __init__(self, model: nn.Module, cfg):
        loss_cfg = _cfg_get(cfg, "Loss", {})
        dataset_cfg = _cfg_get(cfg, "Dataset", {})
        head_cfg = _cfg_get(_cfg_get(cfg, "Model", {}), "Head", {})

        self.nc = _cfg_get(dataset_cfg, "nc", 1)
        self.use_dfl = bool(_cfg_get(loss_cfg, "use_dfl", True))
        self.reg_max = int(_cfg_get(loss_cfg, "reg_max", 16))
        self.qfl_loss_weight = float(_cfg_get(loss_cfg, "qfl_loss_weight", 0.5))
        self.box_loss_weight = float(_cfg_get(loss_cfg, "box_loss_weight", 7.5))
        self.dfl_loss_weight = float(_cfg_get(loss_cfg, "dfl_loss_weight", 1.5))
        self.topk = int(_cfg_get(loss_cfg, "topk", 10))
        self.alpha = float(_cfg_get(loss_cfg, "alpha", 1.0))
        self.beta = float(_cfg_get(loss_cfg, "beta", 6.0))

        self.device = next(model.parameters()).device if isinstance(model, nn.Module) else torch.device("cpu")
        self.strides = _cfg_get(head_cfg, "strides", None)
        detect_module = getattr(model, "head", None)
        if self.strides is None and detect_module is not None and hasattr(detect_module, "stride"):
            self.strides = detect_module.stride
        if self.strides is None:
            self.strides = [8, 16, 32]
        if isinstance(self.strides, torch.Tensor):
            self.strides = self.strides.tolist()
        self.strides = [float(s) for s in self.strides]

        self.no = self.nc + self.reg_max * 4
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(self.topk, self.nc, self.alpha, self.beta)
        self.dfl = DFL(self.reg_max) if self.use_dfl else nn.Identity()

    def __call__(
        self,
        preds: Sequence[torch.Tensor],
        targets: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        if isinstance(targets, torch.Tensor):
            batch = targets[:, 0].long()
            imgs = int(batch.max().item() + 1) if batch.numel() else preds[0].shape[0]
            per_img: list[list[torch.Tensor]] = [[] for _ in range(imgs)]
            for item in targets:
                per_img[int(item[0].item())].append(item[1:].to(preds[0].dtype))
            targets = [torch.stack(t) if t else preds[0].new_zeros((0, 5)) for t in per_img]
        batch_size = preds[0].shape[0]
        dtype = preds[0].dtype
        device = preds[0].device

        anchor_points, stride_tensor = make_anchors(list(preds), self.strides, grid_cell_offset=0.5)
        stride_tensor = stride_tensor.to(device=device, dtype=dtype)
        anchor_points = anchor_points.to(device=device, dtype=dtype)
        stride_broadcast = stride_tensor.view(1, -1, 1)
        anchor_points_grid = anchor_points
        anchor_points_pixel = anchor_points * stride_tensor

        x_cat = torch.cat([p.view(batch_size, self.no, -1) for p in preds], dim=2)
        pred_distri = x_cat[:, : self.reg_max * 4, :]
        pred_scores = x_cat[:, self.reg_max * 4 :, :].permute(0, 2, 1).contiguous()

        if self.use_dfl:
            pred_dist = self.dfl(pred_distri).permute(0, 2, 1).contiguous()
        else:
            pred_dist = pred_distri.view(batch_size, 4, -1).permute(0, 2, 1).contiguous()
        pred_bboxes = dist2bbox(pred_dist, anchor_points_grid.unsqueeze(0), xywh=False) * stride_broadcast

        max_gt = max((t.shape[0] for t in targets), default=0)
        gt_labels = torch.full((batch_size, max_gt, 1), self.nc, device=device, dtype=torch.long)
        gt_bboxes = torch.zeros((batch_size, max_gt, 4), device=device, dtype=dtype)
        mask_gt = torch.zeros((batch_size, max_gt, 1), device=device, dtype=dtype)
        for i, t in enumerate(targets):
            if t is None or t.numel() == 0:
                continue
            n = min(t.shape[0], max_gt)
            gt_labels[i, :n, 0] = t[:n, 0].long()
            gt_bboxes[i, :n] = t[:n, 1:]
            mask_gt[i, :n, 0] = 1.0

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(), pred_bboxes.detach(), anchor_points_pixel, gt_labels, gt_bboxes, mask_gt
        )

        num_pos = fg_mask.sum().clamp_(min=1.0)
        pred_scores_pos = pred_scores[fg_mask]
        target_scores_pos = target_scores[fg_mask]
        loss_cls = self.bce(pred_scores_pos, target_scores_pos).sum() if pred_scores_pos.numel() else pred_scores_pos.new_tensor(0.0)
        loss_cls = loss_cls / num_pos

        pred_boxes_pos = pred_bboxes[fg_mask]
        target_boxes_pos = target_bboxes[fg_mask]
        if pred_boxes_pos.numel():
            iou = bbox_iou(pred_boxes_pos, target_boxes_pos)
            loss_box = (1.0 - iou).mean()
        else:
            loss_box = pred_bboxes.sum() * 0.0

        if self.use_dfl and fg_mask.any():
            pred_logits = pred_distri.permute(0, 2, 1).contiguous().view(batch_size, -1, 4, self.reg_max)
            pred_logits_pos = pred_logits[fg_mask]
            target_boxes_grid = target_bboxes / stride_broadcast
            target_dist = bbox2dist(anchor_points_grid, target_boxes_grid, self.reg_max)
            target_dist_pos = target_dist[fg_mask]
            loss_dfl = self.distribution_focal_loss(pred_logits_pos, target_dist_pos)
        else:
            loss_dfl = pred_bboxes.sum() * 0.0

        total_loss = (
            self.box_loss_weight * loss_box
            + self.qfl_loss_weight * loss_cls
            + self.dfl_loss_weight * loss_dfl
        )

        loss_dict = {
            "loss_cls": self.qfl_loss_weight * loss_cls,
            "loss_iou": self.box_loss_weight * loss_box,
            "loss_dfl": self.dfl_loss_weight * loss_dfl,
            "loss": total_loss,
            "num_fg": fg_mask.sum(),
        }
        return total_loss, loss_dict

    def distribution_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(min=0, max=self.reg_max - 1 - 1e-4)
        floor = target.floor().long()
        ceil = (floor + 1).clamp_(max=self.reg_max - 1)
        weight_ceil = target - floor.float()
        weight_floor = 1.0 - weight_ceil
        pred = pred.view(-1, self.reg_max)
        floor_loss = F.cross_entropy(pred, floor.view(-1), reduction="none")
        ceil_loss = F.cross_entropy(pred, ceil.view(-1), reduction="none")
        mask = (target.view(-1) + 1 < self.reg_max).float()
        loss = floor_loss * weight_floor.view(-1) + ceil_loss * weight_ceil.view(-1) * mask
        return loss.mean()


Loss = ComputeTalLoss

__all__ = ["ComputeTalLoss", "Loss"]
