import pandas as pd
import time
from AutoEncoder import AutoEncoder
import numpy as np
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from lightcurve import LightCurve
import lightdataset
from skelm import ELMClassifier


def classify(types, dropout, batch_size, hidden_size, output, modelname):
    model = AutoEncoder(1, output, hidden_size, 2, 400, dropout).cuda()
    model.load_state_dict(torch.load(modelname))
    dataset = lightdataset.LightDataSet('./data/train_set')
    loader = lightdataset.DataLoader(dataset, batch_size, shuffle=True)
    test_dataset = lightdataset.LightDataSet('./data/test_set')
    test_loader = lightdataset.DataLoader(test_dataset, batch_size, shuffle=True)

    train_feature = None
    train_label = np.array([])
    test_feature = None
    test_label = np.array([])

    for idx, (x, y, name) in enumerate(loader):
        # print(idx)
        x = x.cuda()
        out = model.encode(x).cpu()
        train_feature = np.append(train_feature, out.cpu().detach().numpy(), axis=0) if train_feature is not None else out.cpu().detach().numpy()
        train_label = np.append(train_label, y.cpu().detach().numpy())
        # # train_label.extend(y.cpu().detach().numpy())
        # print(train_label,  train_label.shape)

    for idx, (x, y, name) in enumerate(test_loader):
        x = x.cuda()
        out = model.encode(x).cpu()
        test_feature = np.append(test_feature, out.cpu().detach().numpy(), axis=0) if test_feature is not None else out.cpu().detach().numpy()
        test_label = np.append(test_label, y.cpu().detach().numpy())

    drop_train = []
    drop_test = []
    for i in range(len(train_label)):
        if train_label[i] == -1:
            print(i)
            drop_train.append(i)
    for i in range(len(test_label)):
        if train_label[i] == -1:
            drop_test.append(i)
    train_feature=np.delete(train_feature, drop_train, axis=0)
    train_label=np.delete(train_label, drop_train, axis=0)
    test_feature=np.delete(test_feature, drop_test, axis=0)
    test_label=np.delete(test_label, drop_test, axis=0)

    acc = []
    for i in range(1,2000, 100):
        clf = ELMClassifier(n_neurons=i)
        clf.fit(train_feature,train_label)
        train_result = clf.predict(train_feature)
        precision = sum(train_result == train_label) / train_label.shape[0]
        print('Training precision:', precision, i)

        test_result = clf.predict(test_feature)
        test_precision = sum(test_result == test_label) / test_label.shape[0]
        print('Testing precision:', test_precision)

        acc.append([i, precision, test_precision])
    df = pd.DataFrame(acc, columns=['n_neurons', 'train_acc', 'test_acc'])
    df.to_csv(
        './data/acc_svm/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times[' + str(
            times) + ']-outputsize[' + str(output) + ']-[tanh].csv')


if __name__ == '__main__':
    times = 2000
    types = 'ELM'# tanh
    dropout = 0.25
    batch_size = 64
    hidden_size = 64
    output = 256
    modelname = './model/2023-03-31-11-32-types[LSTM]-times[180]-dropout[0.25]-batchsize[64]-hiddensize[64]-outputsize[256]--formal.pt'
    classify(types, dropout, batch_size, hidden_size, output, modelname)