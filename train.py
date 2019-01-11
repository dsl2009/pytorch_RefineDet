import torch
import torch.optim as optim
from losses import MultiBoxLoss,ArmBoxLoss
from refindet import RefinDet
import data_gen
from dsl_data import data_loader_multi
import config
from np_utils import get_loc_conf_new
import numpy as np
import os
from refine_anchor import Refin_anchors
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

arm_loss = ArmBoxLoss()
refin_anchors_model = Refin_anchors()
net = RefinDet(num_anchors=9, cls_num=21)
net.fpn.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)
net.cuda()

criterion = MultiBoxLoss(num_classes=21)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
gen_bdd = data_gen.get_batch(batch_size=config.batch_size, class_name='voc', image_size=config.image_size,
                             max_detect=100)
sch = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    for x in range(10000):
        images, true_box, true_label = next(gen_bdd)
        try:
            loc_t, conf_t = get_loc_conf_new(true_box, true_label, batch_size=8)
        except:
            continue

        images = np.transpose(images, axes=[0 ,3, 1,2])
        loc_targets = torch.from_numpy(loc_t).float()
        cls_targets = torch.from_numpy(conf_t).long()
        images = torch.from_numpy(images).float()
        inputs, loc_targets, cls_targets = torch.autograd.Variable(images.cuda()), torch.autograd.Variable(
            loc_targets.cuda()), torch.autograd.Variable(cls_targets.cuda())
        optimizer.zero_grad()

        arm_loc, arm_logist, odm_loc, odm_logists = net(inputs)
        odm_box_offset = refin_anchors_model(arm_loc, arm_logist, loc_targets)

        arm_box_loss, arm_cls_loss = arm_loss(loc_targets,arm_loc,cls_targets, arm_logist)

        odm_box_loss, odm_cls_loss = criterion(odm_box_offset.detach(), odm_loc,cls_targets, odm_logists)


        loss = odm_box_loss+odm_cls_loss+arm_box_loss+arm_cls_loss

        loss.backward()
        optimizer.step()


        print('train_loss: %.3f arm_box_loss: %.3f arm_cls_loss: %.3f odm_box_loss: %.3f, odm_cls_loss: %.3f'
              %(loss.item(), arm_box_loss.item(), arm_cls_loss.item(), odm_box_loss.item(), odm_cls_loss.item()))



for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    sch.step(epoch)
    if epoch%5==0:
        torch.save(net.state_dict(), str(epoch)+'net.pth' )
