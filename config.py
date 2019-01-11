import math

image_size = [256,256]
batch_size = 8
feature_stride = [4, 8, 16, 32]
anchor_base_size = [16,32,64,128]
anchor_scale = [1, pow(2,1/3),pow(2, 2/3)]

anchors_size = [ [anchor_scale[0]*x,anchor_scale[1]*x,anchor_scale[2]*x ]  for x in anchor_base_size]

aspect_num = [9, 9, 9,9]




