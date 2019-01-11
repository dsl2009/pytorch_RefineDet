from resnetxt import resnet50
from torch import nn
from torch.nn import functional as F
import torch



class Arm_model(nn.Module):
    def __init__(self, num_anchors):
        super(Arm_model, self).__init__()
        self.num_anchors = num_anchors
        self.c1_conv_share = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c2_conv_share = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c3_conv_share = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c4_conv_share = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fpn_conv = [self.c1_conv_share, self.c2_conv_share,self.c3_conv_share,self.c4_conv_share]
        self.conv_logist = nn.Conv2d(in_channels=256, out_channels=2* self.num_anchors, kernel_size=3, stride=1, padding=1)
        self.conv_box_offset = nn.Conv2d(in_channels=256, out_channels=4* self.num_anchors, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        fpn_maps = []
        for ix, fp in enumerate([c1, c2, c3, c4]):
            fpn_maps.append(F.relu(self.fpn_conv[ix](fp)))
        cls_preds = []
        loc_preds = []
        for fp in fpn_maps:
            loc_pred = self.conv_box_offset(fp)
            cls_pred = self.conv_logist(fp)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(fp.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(fp.size(0), -1, 2)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

