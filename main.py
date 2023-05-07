import os
import random
import shutil

import numpy as np
from lightcurve import LightCurve
import lightdataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

from AutoEncoder import AutoEncoder


def pre_train_slit():
    np.random.seed(0)
    test_data_dest = 'data/test_train/test'
    train_data_dest = 'data/test_train/train'
    data_source = 'data/tensor_data'
    cnt = 0
    max_cnt = 1000  # test_size=200 train_size = 800
    # r = random.randint(0, 23)
    r = 4
    k = 5
    print(r)
    n = 0
    for filename in os.listdir(data_source):
        n += 1
        if n % r == 0:
            cnt += 1
            print(data_source + '/' + filename, cnt)
            if cnt % k == 0:
                shutil.copy(data_source + '/' + filename, test_data_dest)
            else:
                shutil.copy(data_source + '/' + filename, train_data_dest)
        if cnt >= max_cnt:
            break
    print(cnt)


def split():
    np.random.seed(0)
    test_data_dest = 'G:\\A1\\test_set'
    train_data_dest = 'G:\\A1\\train_set'
    data_source = 'G:\\A1\\new_tensor_data'
    cnt = 0
    max_cnt = 24000  # test_size=200 train_size = 800
    # r = np.random.choice(np.arange(1, 11), size=3, replace=False)
    r = [1, 3, 6]
    k = 10
    print(r)
    n = 0
    for filename in os.listdir(data_source):
        n += 1
        #print(n)
        if (n % 10 == r[0]) or (n % 10 == r[1]) or (n % 10 == r[2]):
            shutil.copy(data_source + '/' + filename, test_data_dest)
            print(n)
        else:
            shutil.copy(data_source + '/' + filename, train_data_dest)
        if cnt >= max_cnt:
            break
    print(cnt)

def find_you():
    dataset = lightdataset.LightDataSet('data/train_set')
    loader = lightdataset.DataLoader(dataset, 1, shuffle=True)
    for idx, (x, y, name) in enumerate(loader):
        # /print(idx, (x, y, name))
        print(x.size(), name)
    dataset = lightdataset.LightDataSet('data/test_set')
    loader = lightdataset.DataLoader(dataset, 1, shuffle=True)
    print('__________________________________________________________________________')
    for idx, (x, y, name) in enumerate(loader):
        # /print(idx, (x, y, name))
        print(x.size(), name)


if __name__ == '__main__':
    find_you()
