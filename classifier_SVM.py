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
from sklearn import svm
from sklearn import preprocessing


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
        # # train_label.extend(y.cpu().detach().numpy())
        # print(test_label,  test_label.shape)

    scaler = preprocessing.MinMaxScaler()
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.fit_transform(test_feature)

    acc = []
    matrix = []
    #训练部分
    for i in range(200,3000,200):
        clf = svm.SVC(kernel=types, gamma='scale', C=2.7, max_iter=2000, decision_function_shape='ovr')
        clf.fit(train_feature, train_label)

        train_result = clf.predict(train_feature)
        precision = sum(train_result == train_label)/train_label.shape[0]
        print('Training precision:', precision, i/10)

        test_result = clf.predict(test_feature)
        test_precision = sum(test_result == test_label)/test_label.shape[0]
        print('Testing precision:', test_precision)

        acc.append([i, precision, test_precision])
        matrix.append([])
    df = pd.DataFrame(acc, columns=['C', 'train_acc', 'test_acc'])
    df.to_csv('./data/acc_svm/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times['+ str(times) + ']-outputsize[' + str(output) +']-gamma[scale].csv')


    # #与xgboost比较时间
    # tot_t = []
    # for i in range(20):
    #     np.random.seed(i)
    #     start_time = time.time()
    #     clf = svm.SVC(kernel=types, gamma='scale', C=2.7, max_iter=times, decision_function_shape='ovr')
    #     clf.fit(train_feature, train_label)
    #     train_time = time.time()
    #
    #     test_result = clf.predict(test_feature)
    #     test_time = time.time()
    #
    #     tot_t.append([i,train_time-start_time, test_time-train_time])
    #     print('time:', train_time-start_time, test_time-train_time)
    # df = pd.DataFrame(tot_t, columns=['i', 'train_time','test_time'])
    # df.to_csv('./data/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times['+ str(times) + ']-outputsize[' + str(output) +']-gamma[scale]-[svm_time].csv')



if __name__ == '__main__':
    times = 2000
    types = 'rbf'
    dropout = 0.25
    batch_size = 64
    hidden_size = 64
    output = 256
    modelname = './model/2023-03-31-11-32-types[LSTM]-times[180]-dropout[0.25]-batchsize[64]-hiddensize[64]-outputsize[256]--formal.pt'
    classify(types, dropout, batch_size, hidden_size, output, modelname)
