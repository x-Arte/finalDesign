import os
import time

from upsilont import UPSILoNT
from upsilont.features import VariabilityFeatures
import numpy as np
import pandas as pd
from lightcurve import LightCurve, fit_super_smoother
import lightdataset
from lightcurve import get_bigmacc
from sklearn import svm


def save_feature_and_label():
    # Extract features from a set of light-curves.
    train_feature_list = []
    train_label_list = []
    bigmacc = get_bigmacc('data/Stars_Catalog.csv')
    data_source = "G:\\A1\\asassnvarlc_vband_complete-001\\vardb_files"
    right_data_train = "data/train_set"
    right_data_test = "data/test_set"
    cnt = 0

    max_cnt = 100000  # limit the dataset size
    for filename in os.listdir(right_data_test):
        cnt += 1
        if cnt % 1000 == 999:
            print(cnt)
        if cnt >= max_cnt:
            break
        df = pd.read_csv(data_source + '\\' + filename[:filename.rfind('.')] + '.dat', delim_whitespace=True, header=1)

        name = filename[:filename.rfind('.')]
        entry = bigmacc.loc[name]
        period = entry.period

        if not pd.isnull(entry.classified):
            label = entry.variable_type
        elif entry.class_probability > 0.95:
            label = entry.variable_type
        else:
            continue

        df['HJD'], df['MAG'], df['MAG_ERR'], ss_resid = fit_super_smoother(df['HJD'], df['MAG'], df['MAG_ERR'], period)
        # print(ss_resid)
        if ss_resid > 0.7:
            continue
        df['HJD'] %= period
        df.sort_values('HJD', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.drop_duplicates()

        date = np.array(df['HJD'])
        mag = np.array(df['MAG'])
        err = np.array(df['MAG_ERR'])

        var_features = VariabilityFeatures(date, mag, err).get_features()
        train_feature_list.append(var_features)
        train_label_list.append(label)
        print(len(train_feature_list), len(train_label_list))

    features = pd.DataFrame(train_feature_list)
    labels = pd.DataFrame(train_label_list)

    features.to_csv('data/upsilont_test_features.csv')
    labels.to_csv('data/upsilont_test_labels.csv')

def svm_classify():
    # Classify.
    # ut = UPSILoNT()
    # label, prob = ut.predict(feature_list, return_prob=True)
    train_feature_file = pd.read_csv('data/upsilont_train_features.csv')
    train_label_file = pd.read_csv('data/upsilont_train_labels.csv')
    test_feature_file = pd.read_csv('data/upsilont_test_features.csv')
    test_label_file = pd.read_csv('data/upsilont_test_labels.csv')


    train_feature = train_feature_file.to_numpy()
    train_label = np.array([])
    test_feature = test_feature_file.to_numpy()
    test_label = np.array([])

    for i in train_label_file['0']:
        if i == 'EA':
            train_label=np.append(train_label, 0)
        elif i == 'EB':
            train_label=np.append(train_label, 1)
        else:
            train_label=np.append(train_label, 2)
    for i in test_label_file['0']:
        if i == 'EA':
            test_label=np.append(test_label, 0)
        elif i == 'EB':
            test_label=np.append(test_label, 1)
        else:
            test_label=np.append(test_label, 2)

    print(test_label)

    acc = []

    for i in range(200, 3000, 200):
        clf = svm.SVC(kernel='rbf', gamma=0.05, C=5, max_iter=i)
        clf.fit(train_feature, train_label)

        train_result = clf.predict(train_feature)
        precision = sum(train_result == train_label)/train_label.shape[0]
        print('Training precision:', precision, i/10)

        test_result = clf.predict(test_feature)
        test_precision = sum(test_result == test_label)/test_label.shape[0]
        print('Testing precision:', test_precision)

        acc.append([i, precision, test_precision])
    df = pd.DataFrame(acc, columns=['C', 'train_acc', 'test_acc'])
    df.to_csv('./data/acc_svm/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[rbf]-times[3000]-outputsize[' + str(64) +']-gamma[0.05].csv')


if __name__ == '__main__':
    # save_feature_and_label()
    svm_classify()