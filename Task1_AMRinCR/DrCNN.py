'''
@Project ：PytorchTest
@File ：DrCNN.py
@IDE  ：PyCharm
@Author ：Jade
@Date ：2021/11/11 22:44
'''

import torch
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Task1_AMRinCR.load_data  import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1', 'gfsk1', 'pam4', 'QAMs']

#读取数据并对数据进行预处理
#将train数据集(14,300000)按比例7:3分为train和eval两部分
#test数据集(14,6000)用于测试
def load_data(batch_size,path_train,path_test,snr):
    train_data, train_label, eval_data, eval_label = load_train_data_from_mat(path_train, snr)
    test_data, test_label = load_test_data_from_mat(path_test,snr)
    # print("train_data.shape:",train_data.shape)  #(13120, 2, 128, 1) - 300000/128*8*0.7
    # print("eval_data.shape:", eval_data.shape)   #(5624, 2, 128, 1) - 300000/128*8*0.3
    # print("test_data.shape:", test_data.shape)   #(368, 2, 128, 1) - 6000/128*8

    train_data = np.swapaxes(train_data, 3, 1)
    # print("train_data.shape:", train_data.shape)
    train_data = np.swapaxes(train_data, 3, 2)
    # print("train_data.shape:", train_data.shape)
    eval_data = np.swapaxes(eval_data, 3, 1)
    eval_data = np.swapaxes(eval_data, 3, 2)
    # print(eval_data.shape)
    test_data = np.swapaxes(test_data, 3, 1)
    test_data = np.swapaxes(test_data, 3, 2)

    #将数据类型转化为torch网络需要的数据类型
    #并转化为张量
    train_data = torch.from_numpy(train_data.astype(np.float32))
    train_label = torch.from_numpy(train_label)
    eval_data = torch.from_numpy(eval_data.astype(np.float32))
    eval_label = torch.from_numpy(eval_label)
    test_data = torch.from_numpy(test_data.astype(np.float32))
    test_label = torch.from_numpy(test_label)

    #使用TensorDataset将数据整合到一起
    train_iter = Data.TensorDataset(train_data,train_label)
    eval_iter = Data.TensorDataset(eval_data,eval_label)
    # test_iter = Data.TensorDataset(test_data,test_label)

    #对数据集进行批量处理
    train_loader = Data.DataLoader(
        dataset = train_iter,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 1
    )
    eval_loader = Data.DataLoader(
        dataset=eval_iter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    # test_loader = Data.DataLoader(
    #     dataset=test_iter,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=1
    # )
    return train_loader, eval_loader, test_data,test_label


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
            nn.Linear(32, 7),
            # train中的损失函数CrossEntropyLoss包括了softmax + log + NLLLoss
            # 所以不需要softmax
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.hidden1(x)
        output = self.hidden2(x.view(x.shape[0], -1))
        return output

def train(net, train_loader, eval_loader, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
    # 损失函数
    loss = torch.nn.CrossEntropyLoss()
    #将eval_acc加入列表中，方便进行可视化
    eval_acc_all = []
    for epoch in range(num_epochs):
        train_l_sum = 0.0  #只计算train_loss是无意义的，需对比test_loss与train_loss
        train_acc_sum, n, batch_count = 0.0, 0, 0
        for X, y in train_loader:
            #分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
            X = X.float().to(device)
            # y = y.float().to(device)
            #分类问题中，pytorch默认的预测标签是64位有符号整型数据
            y = y.long().to(device)
            output = net(X)
            l = loss(output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 训练loss
            # train_l_sum += l.cpu().item()
            train_acc_sum += (output.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        eval_acc,eval_loss_sum = evaluate_accuracy(eval_loader, net)
        eval_acc_all.append(eval_acc)
        train_acc = train_acc_sum / n

        print('epoch %d, loss %.4f, train acc %.3f, eval acc %.3f'
              % (epoch + 1, eval_loss_sum/batch_count, train_acc, eval_acc))
        # 打印测试成功率曲线图
        plt.figure()
        plt.plot(eval_acc_all, "r-")
        plt.ylim(0.0, 1.1)
        plt.title("Test acc per epoch")
        plt.show()

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
        # print(f"device:{device}")  #device:cuda:0
    acc_sum, n = 0.0, 0
    eval_loss_sum = 0.0
    # with torch.no_grad():
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            #获得测试loss
            X = X.float().to(device)
            y = y.long().to(device)
            output = net(X)
            l = torch.nn.functional.cross_entropy(output, y)
            eval_loss_sum += l.cpu().item()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
            # print(net(X.float().to(device)).argmax(dim=1), y.float().to(device))
            net.train()
        n += y.shape[0]
    return acc_sum / n, eval_loss_sum

#测试模型-画出混淆矩阵
def plot_confusion_matrix(test_data,test_label,net):
    test_data = test_data.float().to(device)
    test_label = test_label.long().to(device)
    output = net(test_data)
    # 返回指定维度最大值的序号下标
    pre_lab = torch.argmax(output, 1)
    #test准确度
    test_acc = accuracy_score(test_label.cpu(), pre_lab.cpu())
    print(f"test_acc:{test_acc}")
    conf_mat = confusion_matrix(test_label.cpu(), pre_lab.cpu())
    df_cm = pd.DataFrame(conf_mat, index=radio_classes, columns=radio_classes)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    net = DrCNN()
    # print(net)
    batch_size = 128
    path_train = "../dataset/Test1_dataset/signal_dataset/train"
    path_test = "../dataset/Test1_dataset/signal_dataset/test"
    num_epochs = 80

    #加载数据
    #第3个参数是SNR
    #将train数据集分为验证集和训练集
    train_loader, eval_loader ,test_data,test_label= load_data(batch_size,path_train,path_test,13)
    #训练模型
    train(net, train_loader, eval_loader,device, num_epochs)
    #测试模型-画出混淆矩阵
    plot_confusion_matrix(test_data,test_label,net)