'''
@Project ：PytorchTest 
@File ：wrong_DrCNN.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2021/11/4 20:26 
'''
import torch
import torchvision
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as f
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, DataLoader
from Test1_AMRinCR.wrong_load  import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取数据并对数据进行预处理
def load_data(snr):
    train_data, train_label = load_train_val_data_from_mat(
        '../dataset/Test1_dataset/signal_dataset/train', snr)
    train_data = np.swapaxes(train_data, 3, 1)
    # print("train_data.shape:", train_data.shape)
    train_data = np.swapaxes(train_data, 3, 2)
    # print("train_data.shape:", train_data.shape)

    #将数据类型转化为torch网络需要的数据类型
    #并转化为张量
    train_data = torch.from_numpy(train_data.astype(np.float32))
    train_label = torch.from_numpy(train_label)

    #使用TensorDataset将数据整合到一起
    train_iter = Data.TensorDataset(train_data,train_label)
    # print("train_data.shape:", len(train_iter))
    #对数据集进行批量处理
    train_loader = Data.DataLoader(
        dataset = train_iter,
        batch_size = 128,
        shuffle = False,
        num_workers = 1
    )
    return train_loader

#网络参数初始化
class DrCNN(nn.Module):
    def __init__(self):
        super(DrCNN, self).__init__()
        #2层CNN
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 128, (2, 8), 1),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, (1, 16), stride=1, padding=(0, 4)),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        #4层全连接
        self.hidden2 = nn.Sequential(
            nn.Linear(7296, 128),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 7)
            # nn.Softmax()
        )
        # self.out = nn.Linear(32, 7),

    def forward(self, x):
        x = self.hidden1(x)
        output = self.hidden2(x.view(x.shape[0], -1))
        return output


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.float().to(device)).argmax(dim=1) == y.float().to(device)).float().sum().cpu().item()
                # print(net(X.float().to(device)).argmax(dim=1), y.float().to(device))
                net.train()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_loader, test_loader, batch_size, device, num_epochs):
    net = net.to(device)
    print("training on", device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08)
    # 损失函数
    loss = torch.nn.CrossEntropyLoss()
    #将test_acc加入列表中，方便进行可视化
    test_acc_all = []
    for epoch in range(num_epochs):
        train_l_sum = 0.0
        train_acc_sum, n, batch_count = 0.0, 0, 0
        for X, y in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            y_hat = net(X)
            y = y.long()
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_loader, net)
        test_acc_all.append(test_acc)
        train_acc = train_acc_sum / n
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / batch_count, train_acc, test_acc))
        #打印测试成功率曲线图
        plt.figure()
        plt.plot(test_acc_all, "r-")
        plt.ylim(0.0,1.1)
        plt.title("Test acc per epoch")
        plt.show()

if __name__ == '__main__':
    net = DrCNN()
    # print(net)
    train_loader = load_data(10)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()
    net,train_process = train_model(net,train_loader,0.8,loss_func,optimizer,30)

    # train(net, train_loader, test_loader,batch_size,  device, num_epochs)
