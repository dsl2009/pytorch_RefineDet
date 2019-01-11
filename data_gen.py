import random
import numpy as np
import config
from dsl_data import aug_utils
from dsl_data import xair_guoshu, mianhua, bdd, voc, Lucai,BigLand

def get_batch(batch_size,class_name, is_shuff = True,max_detect = 50,image_size=300, is_rcnn = False):
    if class_name == 'guoshu':
        data_set = xair_guoshu.Tree('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree',
                                    config.image_size)
    elif class_name == 'mianhua':
        data_set = mianhua.MianHua('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/mianhua/open',
                                   config.image_size)
    elif class_name == 'bdd':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=False)
    elif class_name == 'bdd_crop':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=True)
    elif class_name == 'voc':
        data_set = voc.VOCDetection(root='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit',
                                   image_size=config.image_size)
    elif class_name == 'lvcai':
        data_set = Lucai.Lucai(image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/round2', image_size=config.image_size, is_crop=False)

    length = data_set.len()
    idx = list(range(length))


    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, box, lab = data_set.pull_item(idx[index])
            except:
                index = index+1
                if index >= length:
                    index = 0
                continue

            if  img is None or len(lab) == 0 or len(lab)>100:
                index+=1
                if index >= length:
                    index = 0
                continue
            if is_rcnn:
                lab = np.asarray(lab)+1
            if False:

                if random.randint(0,1)==1:
                   img, box = aug_utils.fliplr_left_right(img,box)
                if random.randint(0,1)==1:
                   img, box = aug_utils.fliplr_up_down(img,box)

                img = (img -[123.15, 115.90, 103.06])/255.0

            else:
                img = (img - [123.15, 115.90, 103.06]) / 255.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                index=index+1
                b=b+1
            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                index = index + 1
                b = b + 1
            if b>=batch_size:
                yield [images,boxs,label]
                b = 0
            if index>= length:
                index = 0

def get_batch_mask(batch_size,class_name, is_shuff = True,max_detect = 50,image_size=300,mask_shape =[28,28], is_rcnn = False):
    if class_name == 'guoshu':
        data_set = xair_guoshu.Tree_mask('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/',
                                    config.image_size)
    elif class_name == 'mianhua':
        data_set = mianhua.MianHua('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/mianhua/open',
                                   config.image_size)
    elif class_name == 'bdd':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=False)
    elif class_name == 'bdd_crop':
        data_set = bdd.BDD(js_file='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json',
                           image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train',
                                   image_size=config.image_size, is_crop=True)
    elif class_name == 'voc':
        data_set = voc.VOCDetection(root='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit',
                                   image_size=config.image_size)
    elif class_name == 'lvcai':
        data_set = Lucai.Lucai(image_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/round2', image_size=config.image_size, is_crop=False)
    elif class_name == 'land':
        data_set = data_set = BigLand.BigLandBox(image_size=config.image_size)

    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)
            try:
                img, box, lab, mask = data_set.pull_item(idx[index])
            except:
                index = index+1
                if index >= length:
                    index = 0
                continue

            if  img is None or len(lab) == 0 or len(lab)>100:
                index+=1
                if index >= length:
                    index = 0
                continue
            if is_rcnn:
                lab = np.asarray(lab)+1

            img = (img - [123.15, 115.90, 103.06]) / 255.0
            mask = mask/255.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                #true_mask = np.zeros(shape=[batch_size,max_detect,mask_shape[0], mask_shape[1]],dtype=np.float32)
                true_mask = np.zeros(shape=[batch_size,image_size[0],image_size[1],1], dtype=np.float32)
                label = np.zeros(shape=[batch_size, max_detect], dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                true_mask[b,:,:,0] = mask
                index=index+1
                b=b+1
            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                true_mask[b, :, :, 0] = mask
                index = index + 1
                b = b + 1
            if b>=batch_size:
                yield [images,boxs,label, true_mask]
                b = 0
            if index>= length:
                index = 0