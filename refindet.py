from resnetxt import resnet50
from torch import nn
from torch.nn import functional as F
import torch
from odm_model import Odm_model
from arm_model import Arm_model


class RefinDet(nn.Module):
    def __init__(self, num_anchors, cls_num):
        super(RefinDet, self).__init__()
        self.num_class = cls_num
        self.num_anchors = num_anchors
        self.fpn = resnet50()
        self.arm = Arm_model(num_anchors=num_anchors)
        self.odm = Odm_model(num_anchors=num_anchors,cls_num=cls_num)


    def forward(self, img):
        c1, c2, c3, c4 = self.fpn(img)
        arm_loc, arm_logist = self.arm([c1, c2, c3, c4])
        odm_loc, odm_logist = self.odm([c1, c2, c3, c4])

        return arm_loc, arm_logist, odm_loc, odm_logist





if __name__ == '__main__':
    ig = torch.randn([1, 3, 256, 256])
    md = RefinDet(9, 21)
    arm_loc, arm_logist, odm_loc, odm_logist = md(ig)
    print(odm_logist.size())
