import os

import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
from .Dataset import Data
import sys
sys.path.append('../')
from args import args

def transform():
    return Compose([
        ToTensor(),
    ])

class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input

def get_train_data(traindata_dir):
    return Data(traindata_dir, transform=transform())

def get_eval_data(evaldata_dir):
    return Data(evaldata_dir, transform=transform())

def get_test_data(testdata_dir):
    return Data(testdata_dir, transform=transform())