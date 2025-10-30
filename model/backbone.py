import math
import torch.nn as nn

from modules import C2PSA, C3k2, Conv, SPPF

class Yolo11BackBone(nn.Module):
    """YOLO11 backbone producing P3, P4 and P5 feature maps."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.Model
        backbone_cfg = model_cfg.Backbone

        self.gd = getattr(model_cfg, "depth_multiple", 1.0)
        self.gw = getattr(model_cfg, "width_multiple", 1.0)
        self.max_channels = getattr(
            model_cfg,
            "max_channels",
            1024,
        )
        self.act_name = getattr(backbone_cfg, "activation", "SiLU")

        # Base channel definitions prior to width scaling
        self.channels = {
            "stem_conv": 64,
            "down2": 128,
            "c3_p2": 256,
            "down3": 256,
            "c3_p3": 512,
            "down4": 512,
            "c3_p4": 512,
            "down5": 1024,
            "c3_p5": 1024,
            "sppf": 1024,
            "psa": 1024,
        }

        self._apply_width_scaling()

        # Stem and stage definitions aligned with the official YOLO11 backbone layout
        self.stage1 = Conv(3, self.channels["stem_conv"], 3, 2, act=self._activation())
        self.stage2 = Conv(
            self.channels["stem_conv"], self.channels["down2"], 3, 2, act=self._activation()
        )
        self.stage3 = C3k2(
            self.channels["down2"],
            self.channels["c3_p2"],
            self.get_depth(2),
            False,
            0.25,
        )
        self.stage4 = Conv(
            self.channels["c3_p2"], self.channels["down3"], 3, 2, act=self._activation()
        )
        self.stage5 = C3k2(
            self.channels["down3"],
            self.channels["c3_p3"],
            self.get_depth(2),
            False,
            0.25,
        )
        self.stage6 = Conv(
            self.channels["c3_p3"], self.channels["down4"], 3, 2, act=self._activation()
        )
        self.stage7 = C3k2(
            self.channels["down4"],
            self.channels["c3_p4"],
            self.get_depth(2),
            True,
        )
        self.stage8 = Conv(
            self.channels["c3_p4"], self.channels["down5"], 3, 2, act=self._activation()
        )
        self.stage9 = C3k2(
            self.channels["down5"],
            self.channels["c3_p5"],
            self.get_depth(2),
            True,
        )
        self.sppf = SPPF(self.channels["c3_p5"], self.channels["sppf"], 5)
        self.stage10 = C2PSA(
            self.channels["sppf"],
            self.channels["psa"],
            self.get_depth(2),
        )

        self.out_shape = {
            "C3_size": self.channels["c3_p3"],
            "C4_size": self.channels["c3_p4"],
            "C5_size": self.channels["psa"],
        }

    def forward(self, x):
        x = self.stage1(x)  # 0-P1/2
        x = self.stage2(x)  # 1-P2/4
        x = self.stage3(x)
        x = self.stage4(x)  # 3-P3/8
        p3 = self.stage5(x)
        x = self.stage6(p3)  # 5-P4/16
        p4 = self.stage7(x)
        x = self.stage8(p4)  # 7-P5/32
        x = self.stage9(x)
        x = self.sppf(x)
        p5 = self.stage10(x)
        return p3, p4, p5

    def _make_divisible(self, x, divisor):
        if divisor == 0:
            raise ValueError("Divisor must be non-zero when adjusting channel widths.")
        return max(divisor, int(math.ceil(x / divisor) * divisor))

    def _activation(self):
        name = str(self.act_name).lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name in {"hard_swish", "hardswish", "hard-swish"}:
            return nn.Hardswish()
        return nn.SiLU(inplace=True)

    def get_depth(self, n):
            return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        width = n * self.gw
        if self.max_channels:
            width = min(width, self.max_channels)
        return int(self._make_divisible(width, 8))

    def _apply_width_scaling(self):
        for key, value in self.channels.items():
            self.channels[key] = self.get_width(value)