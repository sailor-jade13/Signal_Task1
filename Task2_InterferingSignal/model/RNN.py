# coding=utf-8
'''
@Project ：Signal_Task
@File ：RNN.py
@IDE  ：PyCharm
@Author ：Jade
@Date ：2021/2/16 16:32

RNN为三维数组
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
class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        """
        :param input_dim: 输入数据的维度
        :param hidden_dim: RNN神经元个数
        :param layer_dim: RNN的层数
        :param output_dim: 隐藏层输出的维度（分类的数量）
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim ##RNN神经元个数
        self.layer_dim = layer_dim ##RNN的层数
        ## RNN
        self.rnn = nn.RNN(input_dim,hidden_dim,layer_dim,batch_first=True,nonlinearity='relu')
        #全连接
        self.fc1 = nn.Linear(hidden_dim,output_dim)
    def forward(self, x):
        ## x:[batch,time_ste,input_dim]
        ## out:[batch,time_step,input_dim]
        ## h_n:[layer_dim,batch,hidden_dim]
        out,h_n = self.rnn(x,None) ## None表示h0会使用全0进行初始化
        ## 选取最后一个时间点的output作为输出
        output = self.fc1(out[:,-1,:])
        return output

if __name__ == '__main__':
    input_dim = 64
    hidden_dim = 128
    layer_dim = 2
    output_dim = 8
    net = RNN(input_dim,hidden_dim,layer_dim,output_dim)
    print(net)

    batch_size = 64
    train_path = "../../dataset/Task2_dataset/train"
    test_path = "../../dataset/Task2_dataset/test"
    num_epochs = 40

    #图1-混淆矩阵
    #加载数据
    #将train数据集分为验证集和训练集
    # train_loader, eval_loader= load_train_data(batch_size,train_path,0)
    # test_loader  = load_test_data(batch_size,test_path,0)
    # # #训练模型
    # net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    # # 画出混淆矩阵
    # plot_confusion_matrix(test_loader,net)
    # # 测试模型
    # test_accuracy1(test_loader, net)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    test_acc_all = []
    JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    for i in range(0,14):
        train_loader, eval_loader= load_train_data(batch_size,train_path,i)
        test_data,test_label = load_test_data(batch_size,test_path,i)
        #训练模型
        net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
        #测试准确度
        test_acc = test_accuracy1(test_data,test_label,net)
        test_acc_all.append(test_acc)
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

