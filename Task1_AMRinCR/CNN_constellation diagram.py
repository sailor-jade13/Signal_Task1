'''
@Project ：PytorchTest 
@File ：CNN_constellation diagram.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2021/11/16 10:59
'''
import torch
import torchvision
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import copy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Test1_AMRinCR.load_data import *
from sklearn.metrics import accuracy_score
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Test1_AMRinCR.load_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# QAM信号分类
diagram_classes = ['16qam', '64qam']


def load_data(batch_size, path1, path2, snr):
    train_data, train_label = load_data_from_jpg(path1, snr)
    test_data, test_label = load_data_from_jpg(path2, snr)

    train_data = np.swapaxes(train_data, 3, 1)
    train_data = np.swapaxes(train_data, 3, 2)
    test_data = np.swapaxes(test_data, 3, 1)
    test_data = np.swapaxes(test_data, 3, 2)
    # print(train_data.shape)
    # print(test_data.shape)

    # 把取值范围为[0,255]的PIL图像转换为形状为[C,H,W],取值范围是[0,1.0]的张量
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    train_iter = Data.TensorDataset(train_data, train_label)
    test_iter = Data.TensorDataset(test_data, test_label)

    train_loader = Data.DataLoader(
        dataset=train_iter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_loader = Data.DataLoader(
        dataset=test_iter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    # print(len(train_loader))
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 1),
            nn.PReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(7200, 1024),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        output = self.fc(x.view(x.shape[0], -1))
        return output


def evaluate_accuracy(data_loader, net):
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in data_loader:
            if isinstance(net, torch.nn.Module):
                net.eval()
                X = X.float().to(device)
                y = y.float().to(device)
                output = net(X)
                y = y.long()
                l = nn.functional.cross_entropy(output, y)
                loss_sum += l.cpu().item()
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
                net.train()
            n += y.shape[0]
            # print(n)
    return acc_sum / n


def train(net, train_loader, test_loader, device, num_epoches):
    net = net.to(device)
    # print("training on cuda")
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 记录不同snr下的acc
    acc_snr = 0.0
    # 复制模型的参数
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(num_epoches):
        n, batch_count, train_acc_sum = 0, 0, 0.0
        for X, y in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            output = net(X)
            # print(output.shape)
            # torch.Size([64, 2]) torch.Size([36, 2])-100个数据被分为2个batch
            y = y.long()
            l = loss(output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # (output.argmax(dim=1)).shape = 64 / 36
            # output.argmax(dim=1)的值为0或1
            train_acc_sum += (output.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_acc = train_acc_sum / n
        if train_acc > acc_snr:
            acc_snr = train_acc
        test_acc = evaluate_accuracy(test_loader, net)
        # print('epoch %d, train acc %.3f, test acc %.3f='
        #       % (epoch + 1, train_acc, test_acc))
        # torch.save(net.state_dict(), "lattercnn.pth")
    return acc_snr

if __name__ == '__main__':
    net = CNN()
    path_train = "../dataset/Test1_dataset/constellation_diagram/train"
    path_test = "../dataset/Test1_dataset/constellation_diagram/test"
    batch_size = 64
    num_epoches = 30

    # # 加载数据
    # train_loader, test_loader = load_data(batch_size, path_train, path_test, 0)
    # # 训练模型
    # train(net, train_loader, test_loader, device, num_epoches)

    #记录不同信噪比条件下最好的scc
    acc_snr = []

    for snr in np.arange(14):
        train_loader, test_loader = load_data(batch_size, path_train, path_test, snr)
        acc = train(net, train_loader, test_loader, device, num_epoches)
        acc_snr.append(acc)
        print('snr : %d ,acc : %.3f '% (-8+2*snr,acc))

    # 打印成功率曲线图
    plt.figure()
    plt.plot(acc_snr, "r-")
    plt.ylim(0.0, 1.1)
    plt.title("Acc per snr")
    plt.show()