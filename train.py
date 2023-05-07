import lightdataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from lightcurve import LightCurve
import time

from AutoEncoder import AutoEncoder

def train(adamlr, times, types, dropout, batch_size, hidden_size, output, num_layer):
    start_time = time.time()
    # adamlr = 0.0005
    # times = 180
    # types = 'LSTM'
    # dropout = 0.25
    # batch_size = 64
    # hidden_size = 64
    # output = 128

    dataset = lightdataset.LightDataSet('./data/train_set')
    loader = lightdataset.DataLoader(dataset, batch_size, shuffle=True)
    model = AutoEncoder(1, output, hidden_size, num_layer, 400, dropout).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=adamlr)

    # plt.figure(figsize=(20, 6))
    # plt.scatter(range(len((dataset[0][0]))), dataset[0][0].detach())
    # plt.show()

    test_dataset = lightdataset.LightDataSet('./data/test_set')
    test_loader = lightdataset.DataLoader(test_dataset, batch_size, shuffle=True)
    # for dataline in test_dataset:
    #     if(dataline[1] == -1):
    #         print(dataline[2])
    #     print(dataline[1])

    lo = []

    out = dataset[0][0]
    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(out)), out)
    plt.show()

    for i in range(times):
        lst = []
        lst_test = []
        for idx, (x, y, name) in enumerate(loader):
            # print(idx)
            x = x.cuda()
            opt.zero_grad()
            out = model(x)
            # out = out.squeeze()
            loss = torch.nn.functional.mse_loss(x, out)
            loss.backward()
            opt.step()
            lst.append(loss.item())
        for idx, (x, y, name) in enumerate(test_loader):
            x = x.cuda()
            opt.zero_grad()
            out = model(x)
            # out = out.squeeze()
            loss = torch.nn.functional.mse_loss(x, out)
            lst_test.append(loss.item())
        lo.append([i, sum(lst) / len(lst), sum(lst_test) / len(lst_test)])
        print('epoch:', i + 1, sum(lst) / len(lst), sum(lst_test) / len(lst_test))

        if i % 5 == 4:
            with torch.no_grad():
                out = model(dataset[0][0].cuda()).cpu().detach()
                plt.figure(figsize=(20, 6))
                plt.scatter(range(len(out)), out)
                plt.show()

    torch.save(model.state_dict(),
               'model/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times[' + str(
                   times) + ']-dropout[' + str(dropout) + ']-batchsize[' + str(batch_size) + ']-hiddensize[' + str(
                   hidden_size) + ']-outputsize[' + str(output) + ']-numlayer[' + str(num_layer)+']--pretrain.pt')
    df = pd.DataFrame(lo, columns=['epoch', 'loss', 'test_loss'])
    df.to_csv(
        'data/train_loss/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '-types[' + types + ']-times[' + str(
            times) + ']-dropout[' + str(dropout) + ']-batchsize[' + str(batch_size) + ']-hiddensize[' + str(
            hidden_size) + ']-outputsize[' + str(output)+ ']-numlayer[' + str(num_layer) +']--formal.csv')

    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    # for i in range(2):
        adamlr = 0.0005
        times = 180
        types = 'LSTM'
        dropout = 0.25
        batch_size = 64
        hidden_size = 64
        output = 256
        num_layer = 2
        train(adamlr, times, types, dropout, batch_size, hidden_size, output, num_layer)
