import random
import torch
import os
from torch.utils.data import Dataset, DataLoader
from lightcurve import LightCurve


class LightDataSet(Dataset):
    def __init__(self, path, shuffle=False):
        super(LightDataSet, self).__init__()
        if not os.path.isdir(path):
            raise NotADirectoryError
        self.root = path
        self.ls = os.listdir(path)
        if shuffle:
            random.shuffle(self.ls)

    def __getitem__(self, idx):
        f = torch.load(os.path.join(self.root, self.ls[idx]))
        label = -1
        if f.label == 'EA':
            label = 0
        elif f.label == 'EB':
            label = 1
        elif f.label == 'EW':
            label = 2
        return f.t, label, f.name

    def __len__(self):
        return self.ls.__len__()



