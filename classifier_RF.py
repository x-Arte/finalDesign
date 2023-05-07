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
from sklearn.ensemble import RandomForestClassifier



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

    # criterion = ['gini', 'entropy']
    # ntrees = [50, 100, 250]
    # max_features = [3, 12, 16, 18]
    # min_samples_leaf = [1, 2, 3]
    #
    # parameters = {'criterion': criterion,
    #               'max_features': max_features,
    #               'min_samples_leaf': min_samples_leaf,
    #               'n_estimators': ntrees}
    #               # 'tree_method': 'gpu_hist'}
    #
    # rfmodel = RandomForestClassifier()
    #
    # ## 进行网格搜索
    # print('start!!')
    # clf = GridSearchCV(rfmodel, parameters, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, error_score='raise')
    # clf = clf.fit(train_feature, train_label)
    # print(clf.best_params_)

    # ##正式训练
    parameters = {'criterion': 'gini',
                'max_features': 18,
                'min_samples_leaf': 1,
                'n_estimators': 250}

    clf = RandomForestClassifier(criterion='gini', max_features=1, min_samples_leaf=1, n_estimators=250)
    # 在训练集上训练XGBoost模型
    clf.fit(train_feature, train_label)

    train_predict = clf.predict(train_feature)
    test_predict = clf.predict(test_feature)

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of train:', accuracy_score(train_label, train_predict))
    print('The accuracy of test:', accuracy_score(test_label, test_predict))


if __name__ == '__main__':
    times = 2000
    types = 'rf'
    dropout = 0.25
    batch_size = 64
    hidden_size = 64
    output = 256
    modelname = './model/2023-03-31-11-32-types[LSTM]-times[180]-dropout[0.25]-batchsize[64]-hiddensize[64]-outputsize[256]--formal.pt'
    classify(types, dropout, batch_size, hidden_size, output, modelname)
