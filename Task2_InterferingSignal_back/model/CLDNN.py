'''
@Project ：Signal_Task 
@File ：CLDNN.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/2/23 22:08
'''

import copy
import seaborn as sns
from matplotlib import pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from Task2_InterferingSignal.process.load_data import *
from Task2_InterferingSignal.process.functions import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
# 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

#网络参数初始化
class CLDNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        """
        :param input_dim: 输入数据的维度
        :param hidden_dim: LSTM神经元个数
        :param layer_dim: LSTM的层数
        :param output_dim: 隐藏层输出的维度（分类的数量）
        """
        super(CLDNN, self).__init__()
        # 2层CNN
        self.cnn = nn.Sequential(
            # 在BN层之前将conv2d的偏置bias设置为false
            nn.Conv2d(1, 128, (1, 3), 1, bias=False),
            # BN层输入通道数应与前一层Conv2d的输出通道数一致
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.Dropout(0.5),
            # 在BN层之前将conv2d的偏置bias设置为false
            nn.Conv2d(128, 64, (1, 5), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.Dropout(0.5)
            # nn.Conv2d(64, 32, (1, 7), stride=1, padding=(0, 1), bias=False),
            # nn.BatchNorm2d(32),
            # nn.PReLU(),
            # nn.Conv2d(32, 16, (1, 7), stride=1, padding=(0, 1), bias=False),
            # nn.BatchNorm2d(16),
            # nn.PReLU(),
        )

        #1层LSTM
        self.hidden_dim = hidden_dim ##LSTMNet神经元个数
        self.layer_dim = layer_dim ##LSTMNet的层数
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
        # 2层DNN
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.PReLU(),
            nn.Linear(32,output_dim),
            nn.PReLU(),
            # train中的损失函数CrossEntropyLoss包括了softmax + log + NLLLoss
            # 所以不需要softmax
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        ## r_out:[batch,time_step,output_size]
        ## h_n:[n_layers,batch,hidden_size]
        ## h_c:[n_layers,batch,hidden_size]
        x = self.cnn(x)
        x = torch.squeeze(x, dim=2)
        r_out,(h_n,h_c) = self.lstm(x,None) ## None表示hidden state会0初始化
        ## 选取最后一个时间点的output作为输出
        output = self.fc(r_out[:,-1,:])
        return output


if __name__ == '__main__':
    input_dim = 60
    hidden_dim = 128
    layer_dim = 1
    output_dim = 8
    net = CLDNN(input_dim,hidden_dim,layer_dim,output_dim)
    print(net)

    batch_size = 64
    train_path = "../../dataset/Task2_dataset/train"
    test_path = "../../dataset/Task2_dataset/test"
    num_epochs = 40

    #图1-混淆矩阵
    #加载数据
    #将train数据集分为验证集和训练集
    # train_loader, eval_loader= load_train_data(batch_size,train_path,13)
    # test_loader  = load_test_data(batch_size,test_path,13)
    # # #训练模型
    # my_convnet,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    # # 画出混淆矩阵
    # plot_confusion_matrix(test_loader,net)
    # # 测试模型
    # test_accuracy(test_loader, net)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    test_acc_all = []
    JSR = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    for i in range(0, 14):
        print(f"JSR:{-8 + i * 2}")
        train_loader, eval_loader = load_train_data(batch_size, train_path, i)
        test_loader = load_test_data(batch_size, test_path, i)
        # 训练模型
        net, train_process = train(net, train_loader, eval_loader, device, num_epochs)
        # 测试准确度
        # test_acc = test(test_loader, net)
        test_acc = test_accuracy(test_loader, net)
        test_acc_all.append(test_acc)
        print(test_acc_all)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    plt.figure()
    plt.plot(JSR, test_acc_all, "ro-", label="process acc")
    plt.legend(bbox_to_anchor=(1.00, 0.1))
    # 加网格线
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.xlabel("JSR")
    plt.ylabel("Accuracy")
    plt.title("Average Classification Accuracy")
    plt.show()