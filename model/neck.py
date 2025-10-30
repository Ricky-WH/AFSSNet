import math
import torch.nn as nn

from modules import C3k2, Concat, Conv


class Yolo11Neck(nn.Module):
    """YOLO11 neck producing multi-scale feature maps (P3, P4, P5)."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.Model
        neck_cfg = getattr(model_cfg, "Neck", None)

        self.gd = getattr(model_cfg, "depth_multiple", 1.0)
        self.gw = getattr(model_cfg, "width_multiple", 1.0)
        self.max_channels = getattr(
            model_cfg,
            "max_channels",
            getattr(model_cfg, "max_channel", None),
        )
        self.act_name = getattr(neck_cfg, "activation", "SiLU") if neck_cfg else "SiLU"

        # Base channel definitions prior to width scaling
        self.channels = {
            "input_p3": 512,
            "input_p4": 512,
            "input_p5": 1024,
            "fused_p4": 512,
            "out_p3": 256,
            "down_p4": 256,
            "out_p4": 512,
            "down_p5": 512,
            "out_p5": 1024,
        }

        self._apply_width_scaling()

        c = self.channels
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_p4 = C3k2(
            c["input_p5"] + c["input_p4"],
            c["fused_p4"],
            self.get_depth(2),
            False,
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_p3 = C3k2(
            c["fused_p4"] + c["input_p3"],
            c["out_p3"],
            self.get_depth(2),
            False,
        )

        self.down_p4 = Conv(c["out_p3"], c["down_p4"], 3, 2, act=self._activation())
        self.c3k2_out_p4 = C3k2(
            c["down_p4"] + c["fused_p4"],
            c["out_p4"],
            self.get_depth(2),
            False,
        )

        self.down_p5 = Conv(c["out_p4"], c["down_p5"], 3, 2, act=self._activation())
        self.c3k2_out_p5 = C3k2(
            c["down_p5"] + c["input_p5"],
            c["out_p5"],
            self.get_depth(2),
            True,
        )

        self.concat = Concat()

    def forward(self, inputs):
        p3, p4, p5 = inputs
        x = self.upsample1(p5)
        x = self.concat([x, p4])
        p4_fused = self.c3k2_p4(x)

        x = self.upsample2(p4_fused)
        x = self.concat([x, p3])
        p3_out = self.c3k2_p3(x)

        x = self.down_p4(p3_out)
        x = self.concat([x, p4_fused])
        p4_out = self.c3k2_out_p4(x)

        x = self.down_p5(p4_out)
        x = self.concat([x, p5])
        p5_out = self.c3k2_out_p5(x)

        return [p3_out, p4_out, p5_out]
    
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