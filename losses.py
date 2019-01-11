# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config

def log_sum_exp(x):

    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes):
        super(MultiBoxLoss, self).__init__()

        self.num_classes = num_classes


    def forward(self,loc_t,loc_data,conf_t, conf_data):
        num = loc_data.size(0)
        pos = conf_t > 0
        regard = conf_t ==-1
        conf_t[regard] = 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)


        batch_conf = conf_data.view(-1, self.num_classes)


        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        loss_c[pos.view(-1,1)] = 0
        loss_c[regard.view(-1,1)] = 0

        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)


        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

class ArmBoxLoss(nn.Module):

    def __init__(self, num_classes=2):
        super(ArmBoxLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self,loc_t,loc_data,conf_t, conf_data):
        num = loc_data.size(0)
        pos = conf_t > 0
        regard = conf_t ==-1
        conf_t[regard] = 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        cls_loss = []

        for x in range(config.batch_size):
            lbs = conf_t[x]
            prd_conf = conf_data[x]

            lbs[lbs>0] = 1
            pos_index = lbs>0
            pos_num = pos_index.sum()
            neg_index = lbs==0

            pos_label = lbs[pos_index]
            pos_pred = prd_conf[pos_index]

            neg_label = lbs[neg_index]
            neg_pred = prd_conf[neg_index]

            k = torch.randperm(neg_pred.size(0))
            idx = k[:pos_num*3]

            neg_label = neg_label[idx]
            neg_pred = neg_pred[idx]

            true_label = torch.cat([pos_label, neg_label],0)
            pred_label = torch.cat([pos_pred, neg_pred],0)

            loss_c = F.cross_entropy(pred_label, true_label, size_average=False)
            cls_loss.append(loss_c)

        totoal = sum(cls_loss)

        N = num_pos.data.sum().float()
        loss_l /= N
        lossc= totoal/N
        return loss_l,lossc
if __name__ == '__main__':
    a = torch.randn(4, 5)
    print(a)
    a[a>0] = 1
    print(a)
