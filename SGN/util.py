# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os.path as osp

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./SGN/pretrained/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./SGN/pretrained/')
    else:
        output_dir = os.path.join('./SGN/pretrained/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset, case):
    if case == 0: 
        return 2
    elif case == 1:
        return 120
    elif case == 2:
        return 106
    else:
        return 2
    # if dataset == 'NTU':
    #     return 2
    # elif dataset == 'NTU120':
    #     return 2

    
