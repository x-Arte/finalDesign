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
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier



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

    # learning_rate = [0.1, 0.3, 0.6]
    # subsample = [0.6, 0.8, 0.9]
    # colsample_bytree = [0.6, 0.8, 1.0]
    # max_depth = [3, 5, 8, 10]
    #
    # parameters = {'learning_rate': learning_rate,
    #               'subsample': subsample,
    #               'colsample_bytree': colsample_bytree,
    #               'max_depth': max_depth}
    #               # 'tree_method': 'gpu_hist'}
    #
    # xgbmodel = XGBClassifier(n_estimators=50)
    #
    # ## 进行网格搜索
    # clf = GridSearchCV(xgbmodel, parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1, error_score='raise')
    # clf = clf.fit(train_feature, train_label)
    # print(clf.best_params_)



    # 正式训练
    # parameters = {'learning_rate': 0.3,
    #               'subsample': 0.8,
    #               'colsample_bytree': 0.6,
    #               'max_depth': 8,
    #               'tree_method': 'gpu_hist'}
    #
    # clf = XGBClassifier(parameters)
    # # 在训练集上训练XGBoost模型
    # clf.fit(train_feature, train_label)
    #
    # train_predict = clf.predict(train_feature)
    # test_predict = clf.predict(test_feature)
    #
    # ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    # print('The accuracy of train:', accuracy_score(train_label, train_predict))
    # print('The accuracy of test:', accuracy_score(test_label, test_predict))

    #与xgboost比较时间
    tot_t = []
    for i in range(20):
        np.random.seed(i)
        start_time = time.time()
        parameters = {'learning_rate': 0.3,
                      'subsample': 0.8,
                      'colsample_bytree': 0.6,
                      'max_depth': 8,
                      'tree_method': 'gpu_hist'}

        clf = XGBClassifier(parameters)
        clf.fit(train_feature, train_label)
        train_time = time.time()

        test_result = clf.predict(test_feature)
        test_time = time.time()

        tot_t.append([i, train_time - start_time, test_time - train_time])
        print('time:', train_time - start_time, test_time - train_time)
    df = pd.DataFrame(tot_t, columns=['i', 'train_time', 'test_time'])
    df.to_csv('./data/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times[' + str(
        times) + ']-outputsize[' + str(output) + ']-gamma[scale]-[xgboost_time].csv')


if __name__ == '__main__':
    times = 2000
    types = 'rbf'
    dropout = 0.25
    batch_size = 64
    hidden_size = 64
    output = 256
    modelname = './model/2023-03-31-11-32-types[LSTM]-times[180]-dropout[0.25]-batchsize[64]-hiddensize[64]-outputsize[256]--formal.pt'
    classify(types, dropout, batch_size, hidden_size, output, modelname)
