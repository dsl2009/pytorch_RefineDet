from resnetxt import resnet50
from torch import nn
from torch.nn import functional as F
import torch
from odm_model import Odm_model
from arm_model import Arm_model
import np_utils
import config


def decode_box(anchors,pred_loc,variance=None):
    if variance is None:
        variance =[0.1, 0.2]
    boxes = torch.cat((
        anchors[:, :2] + pred_loc[:, :2] * variance[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(pred_loc[:, 2:] * variance[1])), 1)
    return  boxes

def encode_box(true_box,anchors,variance=None):
    '''
    :param true_box: [center_x, center_y, w, h]: 
    :param anchors: [center_x, center_y, w, h]: 
    :param variance: 
    :return: 
    '''

    if variance is None:
        variance =[0.1, 0.2]
    g_cxcy = (true_box[:, :2]- anchors[:, :2])/(variance[0] * anchors[:, 2:])
    g_wh = torch.log(true_box[:,2:]/ anchors[:, 2:])/variance[1]

    return torch.cat([g_cxcy, g_wh],1)



class Refin_anchors(nn.Module):
    def __init__(self):
        super(Refin_anchors, self).__init__()
        self.anchors = torch.tensor(np_utils.gen_ssd_anchors_new(config.image_size)).float().cuda()

    def forward(self, arm_loc, arm_logist, loc_t):
        new_box_offset = []
        for x in range(config.batch_size):
            refine_anchors = decode_box(anchors=self.anchors, pred_loc=arm_loc[x])
            true_box = decode_box(anchors=self.anchors, pred_loc=loc_t[x])
            new_box_offset.append(encode_box(true_box=true_box, anchors=refine_anchors))
        return torch.stack(new_box_offset,0)


