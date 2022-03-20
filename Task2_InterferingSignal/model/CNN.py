# coding=utf-8
'''
@Project ：Signal_Task
@File ：CNN.py
@IDE  ：PyCharm
@Author ：Jade
@Date ：2021/12/22 16:32
'''
import copy
import seaborn as sns
from matplotlib import pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import torch.nn.functional as F
from pylab import xticks
from Task2_InterferingSignal.process.functions import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
# 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

#网络参数初始化
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #2层CNN
        self.hidden1 = nn.Sequential(
            # 在BN层之前将conv2d的偏置bias设置为false
            nn.Conv2d(1, 128, (1, 3), 1,bias=False),
            #BN层输入通道数应与前一层Conv2d的输出通道数一致
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.Dropout(0.5),
            # 在BN层之前将conv2d的偏置bias设置为false
            nn.Conv2d(128, 64, (1, 5), stride=1, padding=(0, 1),bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.Dropout(0.5)
            nn.Conv2d(64, 32, (1, 7), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, (1, 7), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )

        #1层全连接
        self.hidden2 = nn.Sequential(
            nn.Linear(832,128),
            nn.PReLU(),
            # nn.Dropout(0.5)
            nn.Linear(128,64),
            #nn. Dropout(0.5)
            nn. PReLU(),
            nn.Linear(64,8),
            nn.PReLU(),
            # train中的损失函数CrossEntropyLoss包括了softmax + log + NLLLoss
            # 所以不需要softmax
            # nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.hidden1(x)
        output = self.hidden2(x.view(x.shape[0], -1))
        return output

if __name__ == '__main__':
    net = CNN()
    # print(net)
    batch_size = 64
    train_path = "../../dataset/Task2_dataset/train"
    test_path = "../../dataset/Task2_dataset/test"

    num_epochs = 40

    #图1-混淆矩阵
    #加载数据
    #将train数据集分为验证集和训练集
    # train_loader, eval_loader= load_train_data(batch_size,train_path,13)
    # test_loader = load_test_data(batch_size,test_path,13)
    # #训练模型
    # net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    # # #测试模型-画出混淆矩阵
    # plot_confusion_matrix(test_loader,net)
    # test_accuracy1(test_loader, net)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    test_acc_all = []
    JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    for i in range(0,14):
        train_loader, eval_loader= load_train_data(batch_size,train_path,i)
        test_loader = load_test_data(batch_size,test_path,i)
        #训练模型
        net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
        #测试准确度
        test_acc = test_accuracy1(test_data,test_label,net)
        test_acc_all.append(test_acc)
        print(test_acc_all)
    plt.figure()
    plt.plot(JSR,test_acc_all, "ro-",label="process acc")
    plt.legend(bbox_to_anchor=(1.00,0.1))
    #加网格线
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.xlabel("JSR")
    plt.ylabel("Accuracy")
    plt.title("Average Classification Accuracy")
    plt.show()

