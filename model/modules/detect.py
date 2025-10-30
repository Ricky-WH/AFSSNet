import torch
import torch.nn as nn
import math
from modules import Conv, DWConv, DFL
from utils.tal import make_anchors, dist2bbox

class AFDetect(nn.Module):
    export = False
    format = None
    anchors = torch.empty(0) 
    strides = torch.empty(0)
    legacy=False
    def __init__(self, nc, nl, ch=(), use_fusion=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max=16
        self.no = nc+ 4 * self.reg_max
        self.use_fusion = use_fusion
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # regression branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # classification branch
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        # Classification branch
        # self.cv3 = nn.ModuleList(
        #     nn.Sequential(
        #         Conv(x, c3, 3),
        #         Conv(c3, c3, 3),
        #         nn.Conv2d(c3, self.nc, 1)
        #     ) for x in ch
        # )

        # # Regression branch
        # self.cv2 = nn.ModuleList(
        #     nn.Sequential(
        #         Conv(x, c2, 3),
        #         Conv(c2, c2, 3),
        #         nn.Conv2d(c2, 4 * self.reg_max, 1)
        #     ) for x in ch
        # )


        if self.use_fusion:
            self.cv3_attn = nn.ModuleList(
                nn.Sequential(
                    Conv(c3, c2, 1),
                    nn.Sigmoid()
                ) for _ in ch
            )
            self.cv2_attn = nn.ModuleList(
                nn.Sequential(
                    Conv(c2, c3, 1),
                    nn.Sigmoid()
                ) for _ in ch
            )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    def forward(self, x):
        if self.use_fusion:
            cls_feat = self.cv3[i][:-1](x[i])
            reg_feat = self.cv2[i][:-1](x[i])
                
            cls_attn = self.cv3_attn[i](cls_feat)
            reg_attn = self.cv2_attn[i](reg_feat)
                
            cls_out = self.cv3[i][-1](cls_feat * reg_attn)
            reg_out = self.cv2[i][-1](reg_feat * cls_attn)
        else:
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
        
    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        if self.export and self.format == "imx":
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        return torch.cat((dbox, cls.sigmoid()), 1)
    
    # def bias_init(self):
    #     """Initialize biases for better training stability."""
    #     for m in self.cv3:
    #         final_layer = m[-1]
    #         final_layer.bias.data[:] = math.log(5 / self.nc / (640 / 32) ** 2)
    #     for m in self.cv2:
    #         final_layer = m[-1]
    #         final_layer.bias.data[:] = 1.0
    
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )