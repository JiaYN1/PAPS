import torch, os
import random
import numpy as np
import gdal
import torch.utils.data as data
from PIL import Image, ImageOps
import sys
sys.path.append('../')
from args import args

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['mul.tif'])

def load_img(filepath):
    img = gdal.Open(filepath)
    img = img.ReadAsArray()  # [C, W, H]
    if img.shape[0] == 4:
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float32) / args.max_value  # GF数据集是1023， QB是2047
    return img

def denorm(x):
    x = (x * args.max_value).astype(np.uint16)
    return x

def augment(ms_image, pan_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        pan_image = ImageOps.flip(pan_image)
        # bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            pan_image = ImageOps.mirror(pan_image)
            # bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            # bms_image = bms_image.rotate(180)
            info_aug['trans'] = True
    return ms_image, pan_image, info_aug

class Data(data.Dataset):
    def __init__(self, traindata_dir, transform=None):
        super(Data, self).__init__()

        self.traindata_dir = traindata_dir
        self.transform = transform
        # self.data_augmentation = args.data_augmentation
        # self.normalize = args.normalize
        self.image_filenames = [os.path.join(self.traindata_dir, x.split('_')[0]) for x in os.listdir(self.traindata_dir) if is_image_file(x)]

    def __getitem__(self, index):
        input_pan = load_img('%s_pan.tif' % self.image_filenames[index])
        input_lr = load_img('%s_lr.tif' % self.image_filenames[index])
        input_lr_u = load_img('%s_lr_u.tif' % self.image_filenames[index])
        input_ms = load_img('%s_mul.tif' % self.image_filenames[index])
        # input_pan_d = load_img('%s_pan_d.tif' % self.image_filenames[index])

        if self.transform:
            input_pan = self.transform(input_pan)
            input_lr = self.transform(input_lr)
            input_lr_u = self.transform(input_lr_u)
            input_ms = self.transform(input_ms)
            # input_pan_d = self.transform(input_pan_d)

        return input_pan, input_lr, input_lr_u, input_ms  # , input_pan_d

    def __len__(self):
        return len(self.image_filenames)