from resnetxt import resnet50
from torch import nn
from torch.nn import functional as F
import torch



class Odm_model(nn.Module):
    def __init__(self, num_anchors, cls_num):
        super(Odm_model, self).__init__()
        self.num_class = cls_num
        self.num_anchors = num_anchors
        self.c1_conv_share = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c2_conv_share = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c3_conv_share = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c4_conv_share = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.c4_dconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.c3_dconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.c2_dconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.c1_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c1_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.c2_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c2_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.c3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fpn_conv = [self.c1_conv_share, self.c2_conv_share,self.c3_conv_share,self.c4_conv_share]
        self.conv_logist = nn.Conv2d(in_channels=256, out_channels=self.num_class* self.num_anchors, kernel_size=3, stride=1, padding=1)
        self.conv_box_offset = nn.Conv2d(in_channels=256, out_channels=4* self.num_anchors, kernel_size=3, stride=1, padding=1)




    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        fpn_maps = []
        for ix, fp in enumerate([c1, c2, c3, c4]):
            fpn_maps.append(F.relu(self.fpn_conv[ix](fp)))
        fp1, fp2, fp3, c4 = fpn_maps

        c3 = self.c3_conv2(fp3)
        c3 = c3+self.c4_dconv(c4)
        c3 = F.relu(c3)
        c3 = self.c3_conv3(c3)
        c3 = F.relu(c3)

        c2 = self.c2_conv2(fp2)
        c2 = c2 + self.c3_dconv(fp3)
        c2 = F.relu(c2)
        c2 = self.c2_conv3(c2)
        c2 = F.relu(c2)

        c1 = self.c1_conv2(fp1)
        c1 = c1 + self.c2_dconv(fp2)
        c1 = F.relu(c1)
        c1 = self.c1_conv3(c1)
        c1 = F.relu(c1)

        cls_preds = []
        loc_preds = []
        for fp in [c1, c2, c3, c4]:
            loc_pred = self.conv_box_offset(fp)
            cls_pred = self.conv_logist(fp)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(fp.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(fp.size(0), -1, self.num_class)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

if __name__ == '__main__':
    ig = torch.randn([1,3,256,256])
    md = Odm_model(9, 21)
    loc, log = md(ig)
