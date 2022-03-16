'''
@Project ：Signal_Task 
@File ：ResNet18.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/15 10:45 
'''
# ResNet-18
# 每个模块里有4个卷积层（不计算1×1卷积层），加上最开始的卷积层和最后的全连接层，共计18层

import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import copy
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Task2_InterferingSignal.load_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
#                无干扰 单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

#读取数据并对数据进行预处理
#将train数据集(300000)按比例7:3分为train和eval两部分
def load_train_data(batch_size,path_train,snr):
    train_data, train_label, eval_data, eval_label = load_train_data_from_mat(path_train,snr)
    train_data = torch.unsqueeze(torch.from_numpy(train_data),dim=1)
    eval_data = torch.unsqueeze(torch.from_numpy(eval_data), dim=1)
    train_data = torch.unsqueeze(train_data, dim=2)
    eval_data = torch.unsqueeze(eval_data, dim=2)
    # print("train_data.shape:",train_data.shape)  #(26248,1,1,64) - 300000/64*8*0.7
    # print("eval_data.shape:", eval_data.shape)   #(11248,1,1,64) - 300000/64*8*0.3

    #将数据类型转化为torch网络需要的数据类型
    #并转化为张量
    train_data = train_data.float()
    train_label = torch.from_numpy(train_label)
    eval_data = eval_data.float()
    eval_label = torch.from_numpy(eval_label)

    # #使用TensorDataset将数据整合到一起
    train_iter = Data.TensorDataset(train_data,train_label)
    eval_iter = Data.TensorDataset(eval_data,eval_label)

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
    return train_loader, eval_loader

def load_test_data(batch_size,path_test,snr):
    test_data, test_label = load_test_data_from_mat(path_test,snr)
    test_data = torch.unsqueeze(torch.from_numpy(test_data), dim=1)
    test_data = torch.unsqueeze(test_data, dim=2)
    # print("test_data.shape:", test_data.shape)  # (11248,1,1,64)

    #将数据类型转化为torch网络需要的数据类型
    #并转化为张量
    test_data = test_data.float()
    test_label = torch.from_numpy(test_label)

    test_iter = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(
        dataset=test_iter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return test_loader

# 残差块
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        # 有2个有相同输出通道数的3×3卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 是否使用额外的1×1卷积层来修改通道数
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# 残差网络的模块
def resnet_block(in_channels, out_channels, num_residuals,first_block=False):
    # 第一个模块的通道数同输入通道数一致
    if first_block:
        assert in_channels == out_channels
    blk = []
    # 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels,use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# ResNet18模型
def resnet18(num_classes, in_channels=1):
    # ResNet的前两层：
    # 在输出通道数为64、步幅为2的7×7卷积层后接步幅为2的3×3的最大池化层
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # ResNet使用4个由残差块组成的模块，每个模块使用2个同样输出通道数的残差块。
    # 第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。
    # 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    # 加入全局平均池化层 - 输出: (Batch, 512, 1, 1)
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    # 接上全连接层输出
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net

# 训练过程
def train(net, train_loader, eval_loader, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
    # 损失函数
    loss = torch.nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    eval_loss_all = []
    eval_acc_all = []
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs-1))
        print('-'*20)
        # 每个epoch都分为两个阶段：train和eval
        # #只计算train_loss是无意义的，需对比eval_loss与train_loss
        train_loss  = 0.0
        train_acc = 0
        train_num = 0
        eval_loss = 0.0
        eval_acc = 0
        eval_num = 0
        #训练阶段
        for b_x, b_y in train_loader:
            net.train()
            #分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
            b_x = b_x.float().to(device)
            #分类问题中，pytorch默认的预测标签是64位有符号整型数据
            b_y = b_y.long().to(device)
            output = net(b_x)
            pre_lab = torch.argmax(output,1)
            l = loss(output, b_y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.cpu().item() * b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y.data).cpu().item()
            # train_acc_sum += (output.argmax(dim=1) == y).sum().cpu().item()
            train_num += b_x.size(0)
            # batch_count += 1
        for b_x,b_y in eval_loader:
            net.eval()
            # 分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
            b_x = b_x.float().to(device)
            # 分类问题中，pytorch默认的预测标签是64位有符号整型数据
            b_y = b_y.long().to(device)
            output = net(b_x)
            pre_lab = torch.argmax(output, 1)
            l = loss(output, b_y)
            eval_loss += l.cpu().item() * b_x.size(0)
            eval_acc += torch.sum(pre_lab == b_y.data).cpu().item()
            eval_num += b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_acc/train_num)
        eval_loss_all.append(eval_loss / eval_num)
        eval_acc_all.append(eval_acc/ eval_num)
        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(
            train_loss_all[-1],train_acc_all[-1]) )
        print('Eval Loss: {:.4f} Eval Acc: {:.4f}'.format(
            eval_loss_all[-1], eval_acc_all[-1]))
        if eval_acc_all[-1] > best_acc:
            best_acc = eval_acc_all[-1]
            best_model_wts = copy.deepcopy(net.state_dict())
        # 释放无关内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # 打印测试成功率曲线图
        # plt.figure()
        # plt.plot(eval_acc_all, "r-")
        # plt.ylim(0.0, 1.1)
        # plt.title("Eval acc per epoch")
        # plt.show()
    net.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data = {
            "epoch":range(num_epochs),
            "train_loss_all":train_loss_all,
            "eval_loss_all":eval_loss_all,
            "train_acc_all": train_acc_all,
            "eval_acc_all": eval_acc_all,
        }
    )
    return net,train_process

# 测试模型-法一
def test_accuracy(test_loader,net):
    net.eval()
    test_y_all = torch.LongTensor()
    pre_lab_all = torch.LongTensor()
    for step, (test_data, test_label) in enumerate(test_loader):
        with torch.no_grad():
            test_data = test_data.float().to(device)
            test_label = test_label.long().to(device)
            output = net(test_data)
            # 返回指定维度最大值的序号下标
            pre_lab = torch.argmax(output, 1)
            test_y_all = torch.cat((test_y_all.cpu(), test_label.cpu()))
            pre_lab_all = torch.cat((pre_lab_all.cpu(), pre_lab.cpu()))
    test_acc = accuracy_score(test_y_all, pre_lab_all)
    print(f"test_acc:{test_acc}")
    return test_acc

# 测试模型-法二
def test(test_loader,net):
    net.eval()
    test_acc = 0.0
    test_num = 0
    for step, (b_x,b_y) in enumerate(test_loader):
        with torch.no_grad():
            test_data = b_x.float().cuda()
            test_label = b_y.cuda()
            output = net(test_data)
            pre_lab = output.argmax(dim=1)
            test_acc += torch.eq(pre_lab, test_label).sum().float().item()
            test_num += test_data.shape[0]
    test_acc = test_acc / test_num
    print(f"test_acc:{test_acc}")
    return test_acc

# 画出混淆矩阵
def plot_confusion_matrix(test_loader,net):
    net.eval()
    test_y_all = torch.LongTensor()
    pre_lab_all= torch.LongTensor()
    for step, (b_x, b_y) in enumerate(test_loader):
        with torch.no_grad():
            test_data = b_x.float().to(device)
            test_label = b_y.long().to(device)
            output = net(test_data)
            # 返回指定维度最大值的序号下标
            pre_lab = torch.argmax(output, 1)
            test_y_all = torch.cat((test_y_all.cpu(),test_label.cpu()))
            pre_lab_all = torch.cat((pre_lab_all.cpu(),pre_lab.cpu()))
    conf_mat = confusion_matrix(test_y_all, pre_lab_all)
    df_cm = pd.DataFrame(conf_mat, index=signal_classes, columns=signal_classes)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    net = resnet18(8)
    print(net)

    batch_size = 64
    train_path = "../dataset/Task2_dataset/train"
    test_path = "../dataset/Task2_dataset/test"
    num_epochs = 40

    # 图1-混淆矩阵
    # 加载数据
    # 将train数据集分为验证集和训练集
    # train_loader, eval_loader = load_train_data(batch_size, train_path, 10)
    # test_loader = load_test_data(batch_size, test_path, 10)
    # # 训练模型
    # net, train_process = train(net, train_loader, eval_loader, device, num_epochs)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()
    # # 画出混淆矩阵
    # plot_confusion_matrix(test_loader,net)
    # # 测试模型
    # test_accuracy(test_loader, net)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    test_acc_all = []
    JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    for i in range(0,14):
        print(f"JSR:{-8+i*2}")
        train_loader, eval_loader= load_train_data(batch_size,train_path,i)
        test_loader = load_test_data(batch_size,test_path,i)
        #训练模型
        net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
        #测试准确度
        # test_acc = test(test_loader, net)
        test_acc = test_accuracy(test_loader,net)
        test_acc_all.append(test_acc)
        print(test_acc_all)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    plt.figure()
    plt.plot(JSR,test_acc_all, "ro-",label="Test acc")
    plt.legend(bbox_to_anchor=(1.00,0.1))
    #加网格线
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.xlabel("JSR")
    plt.ylabel("Accuracy")
    plt.title("Average Classification Accuracy")
    plt.show()
