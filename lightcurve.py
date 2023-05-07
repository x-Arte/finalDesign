import pandas
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from supersmoother import SuperSmoother


class LightCurve:
    def __init__(self,
                 name: str,
                 label: str,
                 t: torch.Tensor):
        self.label = label
        self.name = name
        self.t = t


def preprocess(file_path, file_name, index_df, seq_len=200):
    # Read csv and sort by HJD
    # print(file_path + '/' + file_name)
    df = pd.read_csv(file_path + '/' + file_name, delim_whitespace=True, header=1)

    # find period & label
    name = file_name[:file_name.rfind('.')]
    entry = index_df.loc[name]
    period = entry.period

    if not pd.isnull(entry.classified):
        label = entry.variable_type
    elif entry.class_probability > 0.95:
        label = entry.variable_type
    else:
        label = None

    # best_period, best_score = fit_lomb_scargle(df['HJD'], df['MAG'], df['MAG_ERR'])
    # print(period ,best_period, best_score)
    df['HJD'], df['MAG'], df['MAG_ERR'], ss_resid = fit_super_smoother(df['HJD'], df['MAG'], df['MAG_ERR'], period)
    # print(ss_resid)
    if ss_resid > 0.7:
        return None

    #before fold
    # plt.figure(figsize=(20, 6))
    # plt.scatter(df['HJD'], df['MAG'])
    # plt.show()

    # fold
    df['HJD'] %= period
    df.sort_values('HJD', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop_duplicates()

    #after fold
    # plt.figure(figsize=(20, 6))
    # plt.scatter(df['HJD'], df['MAG'])
    # plt.show()

    # Unify the MAG col
    mean = df.mean(axis=0)['MAG']
    std = df.std(axis=0)['MAG']
    df['MAG'] = (df['MAG'] - mean) / std

    # Unify the HJD col
    time_min = df['HJD'][0]
    time_max = df['HJD'].iloc[-1]
    df['HJD'] = (df['HJD'] - time_min) / (time_max - time_min) * seq_len - 0.5

    # Recalculate the MAG with a fixed deltaT
    t0 = 0
    ls = []
    for i in range(seq_len):
        for j in range(t0, len(df)):
            t0 = j
            now_x = df['HJD'][j]
            now_y = df['MAG'][j]
            if now_x > i:
                last_x = df['HJD'][j - 1]
                last_y = df['MAG'][j - 1]
                ls.append(last_y + (i - last_x) * (last_y - now_y) / (last_x - now_x))
                break
            elif now_x == i:
                ls.append(now_y)
                break

    # final data
    # plt.figure(figsize=(20, 6))
    # plt.scatter(range(seq_len), ls)
    # plt.show()

    # Save the new MAG as a tensor
    t = torch.tensor(ls, dtype=torch.float).view(-1, 1)

    return LightCurve(name, label, t)


# def fit_lomb_scargle(times: list, measurements: list, errors: list):
#     from gatspy.periodic import LombScargleFast
#     period_range = (0.005 * (max(times) - min(times)),
#                     0.95 * (max(times) - min(times)))
#     model_gat = LombScargleFast(fit_period=True, silence_warnings=True,
#                                 optimizer_kwds={'period_range': period_range, 'quiet': True})
#     model_gat.fit(times, measurements, errors)
#     best_period = model_gat.best_period
#     best_score = model_gat.score(model_gat.best_period).item()
#     return best_period, best_score


def fit_super_smoother(times: list, measurements: list, errors: list, p: float, periodic: object = True, scale: object = True) -> object:
    model = SuperSmoother(period=p if periodic else None)
    try:
        model.fit(times, measurements, errors)
        ss_resid = np.sqrt(np.mean((model.predict(times) - measurements) ** 2))
        if scale:
            ss_resid /= np.std(measurements)
    except ValueError:
        ss_resid = np.inf
    return times, measurements, errors, ss_resid


def get_bigmacc(path):
    bigmacc_tmp = pd.read_csv(path)
    bigmacc_tmp['asassn_name'] = bigmacc_tmp['asassn_name'].str.replace(' ', '')
    bigmacc_tmp.drop_duplicates(subset=['asassn_name'], keep='first', inplace=True)
    return bigmacc_tmp.set_index(keys='asassn_name')


if __name__ == '__main__':
    bigmacc = get_bigmacc('data/Stars_Catalog.csv')
    # data_source = "G:\\A1\\asas_select_data"
    # data_dest = "G:\\A1\\new_tensor_data"
    data_dest = 'data/tensor_data'
    data_source = 'data/pre_data/test_data'
    cnt = 0

    max_cnt = 100000 # limit the dataset size
    for filename in os.listdir(data_source):
        cnt += 1
        if cnt % 1000 == 999:
            print(cnt)
        if cnt >= max_cnt:
            break
        lc = preprocess(data_source, filename, bigmacc, 400)
        if lc is None:
            continue
        # print(lc.name, lc.label, lc.t)
        torch.save(lc, data_dest + '/' + filename[:filename.rfind('.')] + '.pth')
