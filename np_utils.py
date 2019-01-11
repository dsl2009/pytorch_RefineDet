import numpy as np
import config


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:,2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:,:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]

    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))  # [A,B]

    union = area_a + area_b - inter
    return inter / union  # [A,B]


def over_laps(boxa,boxb):
    A = boxa.shape[0]
    B = boxb.shape[0]
    b_box = np.expand_dims(boxb,0)
    a = np.repeat(boxa, repeats=B, axis=0)
    b = np.reshape(np.repeat(b_box, repeats=A, axis=0), newshape=(-1, 4))
    d = jaccard_numpy(a, b)
    return np.reshape(d,newshape=(A,B))

def pt_from(boxes):
    xy_min = boxes[:, :2] - boxes[:, 2:] / 2
    xy_max = boxes[:, :2] + boxes[:, 2:] / 2
    return np.hstack([xy_min,xy_max])

def pt_from_nms(boxes):
    y_min = boxes[:, 1:2] - boxes[:, 3:] / 2
    y_max = boxes[:, 1:2] + boxes[:, 3:] / 2
    x_min = boxes[:, 0:1] - boxes[:, 2:3] / 2
    x_max = boxes[:, 0:1] + boxes[:, 2:3] / 2

    return np.hstack([y_min,x_min, y_max, x_max])

def encode(matched, priors, variances):

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]

    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.hstack([g_cxcy, g_wh])  # [num_priors,4]


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride=1):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths,box_heights], axis=2).reshape([-1, 2])

    boxes = np.concatenate([box_centers,box_sizes],axis=1)

    # Convert to corner coordinates (y1, x1, y2, x2)
    #boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                           # box_centers + 0.5 * box_sizes], axis=1)


    return boxes


def gen_multi_anchors(scales, ratios, shape, feature_stride, anchor_stride=1):
    anchors = []
    for s in range(len(feature_stride)):
        an = generate_anchors(scales[s],ratios[s],shape[s],feature_stride[s],anchor_stride=1)
        anchors.append(an)
    return np.vstack(anchors)

def gen_ssd_anchors1():
    #scals = [(36,74,96),(136,198,244),(294,349,420)]
    scals = [(24, 32, 64), (96, 156, 244), (294, 349, 420)]
    ratios = [[0.5,1,2],[0.5,1,2],[0.5,1,2]]
    shape =[(64,64),(32,32),(16,16)]
    feature_stride = [8,16,32]
    anchors = gen_multi_anchors(scales=scals,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/512.0
    out = np.clip(anchors, a_min=0.0, a_max=1.0)
    return out




def gen_ssd_anchors_new(image_size):
    size = config.anchors_size
    feature_stride = config.feature_stride
    ratios = [[0.5,1,2]]*len(feature_stride)
    sc = config.anchors_size
    shape = [(image_size[0] / x, image_size[1] / x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc, ratios=ratios, shape=shape, feature_stride=feature_stride)
    anchors = anchors / np.asarray(
        [image_size[1], image_size[0], image_size[1], image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out

def encode_box(true_box, true_label, anchors):
    '''
    :param true_box: xmin ymin xmax ymax 
    :param true_label: 
    :param anchors:  xmin ymin xmax ymax 
    :return: 
    '''
    anchors_point = pt_from(anchors)
    ops = over_laps(true_box, anchors_point)
    best_true = np.max(ops, axis=0)
    best_true_idx = np.argmax(ops, axis=0)

    best_prior = np.max(ops, axis=1)
    best_prior_idx = np.argmax(ops, axis=1)

    for j in range(best_prior_idx.shape[0]):
        best_true_idx[best_prior_idx[j]] = j
        # best_true[best_prior_idx[j]] = 1.0
    matches = true_box[best_true_idx]
    conf = true_label[best_true_idx] + 1
    conf[best_true <= 0.4] = 0
    b1 = best_true > 0.4
    b2 = best_true <= 0.5
    conf[b1 * b2] = -1
    loc = encode(matches, anchors, variances=[0.1, 0.2])
    return conf, loc

def get_loc_conf_new(true_box, true_label,batch_size = 4,cfg = None):
    pri = gen_ssd_anchors_new(config.image_size)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = (true_box_tm[:, 2]- true_box_tm[:,0])*(true_box_tm[:, 3]- true_box_tm[:,1])
        true_box_tm = true_box_tm[np.where(ix > 1e-6)]
        labels = labels[np.where(ix > 1e-6)]
        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
            best_true[best_prior_idx[j]] = 1.0
        matches = true_box_tm[best_true_idx]
        conf = labels[best_true_idx] + 1
        conf[best_true <= 0.4] = 0
        b1 = best_true>0.4
        b2 = best_true<=0.5
        conf[b1*b2] = -1
        loc = encode(matches, pri, variances=[0.1, 0.2])
        loc_t[s] = loc
        conf_t[s] = conf
    return loc_t,conf_t

def test_encode():
    d = gen_ssd_anchors_new(image_size=[256,256])
    box = np.asarray([[0.1,0.1,0.2,0.2],[0.2,0.3,0.3,0.4]])
    lb = np.asarray([0,1])
    encode_box(box, lb, d)



if __name__ == '__main__':
    test_encode()