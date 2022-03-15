# '''
# @Project ：Signal_Task
# @File ：test1.py
# @IDE  ：PyCharm
# @Author ：Jade
# @Date ：2021/12/22 15:40
# '''

import torch
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
from Task2_InterferingSignal.load_data  import *

#
# dat = np.zeros((13120, 2, 128, 1))
# dat = np.swapaxes(dat, 3, 1)
# dat = np.swapaxes(dat, 3, 2)
# print(dat.shape)
#
#
#
# # coding=utf-8
# '''
# @Project ：Signal_Task
# @File ：CNN.py
# @IDE  ：PyCharm
# @Author ：Jade
# @Date ：2021/12/22 16:32
# '''
# import torch
# import copy
# import seaborn as sns
# from matplotlib import pyplot as plt
# import torchvision.transforms as transforms
# import numpy as np
# import torch.utils.data as Data
# import torch.nn as nn
# from sklearn.metrics import accuracy_score,confusion_matrix
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from Task2_InterferingSignal.load_data  import *
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 信号分类
# # 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
# signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']
#
# #读取数据并对数据进行预处理
# #将train数据集(300000)按比例7:3分为train和eval两部分
# def load_train_data(batch_size,path_train):
#     train_data, train_label, eval_data, eval_label = load_train_data_from_mat(path_train)
#     train_data = torch.unsqueeze(torch.from_numpy(train_data),dim=1)
#     eval_data = torch.unsqueeze(torch.from_numpy(eval_data), dim=1)
#     train_data = torch.unsqueeze(train_data, dim=2)
#     eval_data = torch.unsqueeze(eval_data, dim=2)
#     print("train_data.shape:",train_data.shape)  #(26248,1,1,64) - 300000/64*8*0.7
#     print("eval_data.shape:", eval_data.shape)   #(2400,1,1,64) - 300000/64*8*0.3
#
#     #将数据类型转化为torch网络需要的数据类型
#     #并转化为张量
#     train_data = train_data.float()
#     train_label = torch.from_numpy(train_label)
#     eval_data = eval_data.float()
#     eval_label = torch.from_numpy(eval_label)
#
#     # #使用TensorDataset将数据整合到一起
#     train_iter = Data.TensorDataset(train_data,train_label)
#     eval_iter = Data.TensorDataset(eval_data,eval_label)
#
#     #对数据集进行批量处理
#     train_loader = Data.DataLoader(
#         dataset = train_iter,
#         batch_size = batch_size,
#         shuffle = True,
#         num_workers = 1
#     )
#     eval_loader = Data.DataLoader(
#         dataset=eval_iter,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=1
#     )
#     return train_loader, eval_loader
#
# def load_test_data(batch_size,path_test):
#     test_data, test_label = load_test_data_from_mat(path_test)
#     test_data = torch.unsqueeze(torch.from_numpy(test_data), dim=1)
#     test_data = torch.unsqueeze(test_data, dim=2)
#     print("test_data.shape:", test_data.shape)  # (11248,1,1,64)
#
#     #将数据类型转化为torch网络需要的数据类型
#     #并转化为张量
#     test_data = test_data.float()
#     test_label = torch.from_numpy(test_label)
#
#     test_iter = Data.TensorDataset(test_data, test_label)
#     test_loader = Data.DataLoader(
#         dataset=test_iter,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=1
#     )
#     return test_data,test_label
#
# #网络参数初始化
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         #2层CNN
#         self.hidden1 = nn.Sequential(
#             # 在BN层之前将conv2d的偏置bias设置为false
#             nn.Conv2d(1, 128, (1, 5), 1,bias=False),
#             nn.PReLU(),
#             # nn.Dropout(0.5),
#             # 在BN层之前将conv2d的偏置bias设置为false
#             nn.Conv2d(128, 64, (1, 9), stride=1, padding=(0, 2),bias=False),
#             nn.PReLU(),
#             # nn.Dropout(0.5)
#         )
#         #4层全连接
#         self.hidden2 = nn.Sequential(
#             nn.Linear(3584, 128),
#             nn.PReLU(),
#             # nn.Dropout(0.5),
#             nn.Linear(128, 64),
#             nn.PReLU(),
#             # nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.PReLU(),
#             # nn.Dropout(0.5),
#             nn.Linear(32, 8),
#             # train中的损失函数CrossEntropyLoss包括了softmax + log + NLLLoss
#             # 所以不需要softmax
#             # nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.hidden1(x)
#         output = self.hidden2(x.view(x.shape[0], -1))
#         return output
#
# def train(net, train_loader, eval_loader, device, num_epochs):
#     net = net.to(device)
#     print("training on", device)
#     # 优化器
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
#     # 损失函数
#     loss = torch.nn.CrossEntropyLoss()
#     best_model_wts = copy.deepcopy(net.state_dict())
#     #将eval_acc加入列表中，方便进行可视化
#     eval_acc_all = []
#     for epoch in range(num_epochs):
#         train_l_sum = 0.0  #只计算train_loss是无意义的，需对比test_loss与train_loss
#         train_acc_sum, n, batch_count = 0.0, 0, 0
#         for X, y in train_loader:
#             #分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
#             X = X.float().to(device)
#             #分类问题中，pytorch默认的预测标签是64位有符号整型数据
#             y = y.long().to(device)
#             output = net(X)
#             l = loss(output, y)
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             # 训练loss
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (output.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         eval_acc,eval_loss_sum = evaluate_accuracy(eval_loader, net)
#         eval_acc_all.append(eval_acc)
#         train_acc = train_acc_sum / n
#         train_loss = train_l_sum / n
#         print('epoch %d, train loss %.4f, eval loss %.4f, train acc %.3f, eval acc %.3f'
#               % (epoch + 1, train_loss,eval_loss_sum/batch_count, train_acc, eval_acc))
#         # 释放无关内存
#         if hasattr(torch.cuda, 'empty_cache'):
#             torch.cuda.empty_cache()
#         # 打印测试成功率曲线图
#         # plt.figure()
#         # plt.plot(eval_acc_all, "r-")
#         # plt.ylim(0.0, 1.1)
#         # plt.title("Eval acc per epoch")
#         # plt.show()
#
# def evaluate_accuracy(data_iter, net, device=None):
#     if device is None and isinstance(net, torch.nn.Module):
#         device = list(net.parameters())[0].device
#         # print(f"device:{device}")  #device:cuda:0
#     acc_sum, n = 0.0, 0
#     eval_loss_sum = 0.0
#     # 测试函数前加了装饰器，解决了cuda out of memory
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(net, torch.nn.Module):
#                 net.eval()
#                 #获得测试loss
#                 X = X.float().to(device)
#                 y = y.long().to(device)
#                 output = net(X)
#                 l = torch.nn.functional.cross_entropy(output, y)
#                 eval_loss_sum += l.cpu().item()
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
#                 # print(net(X.float().to(device)).argmax(dim=1), y.float().to(device))
#                 net.train()
#             n += y.shape[0]
#     return acc_sum / n, eval_loss_sum
#
# #测试模型-画出混淆矩阵
# def plot_confusion_matrix(test_data,test_label,net):
#     test_data = test_data.float().to(device)
#     test_label = test_label.long().to(device)
#     output = net(test_data)
#     # 返回指定维度最大值的序号下标
#     pre_lab = torch.argmax(output, 1)
#     #test准确度
#     test_acc = accuracy_score(test_label.cpu(), pre_lab.cpu())
#     print(f"test_acc:{test_acc}")
#     conf_mat = confusion_matrix(test_label.cpu(), pre_lab.cpu())
#     df_cm = pd.DataFrame(conf_mat, index=signal_classes, columns=signal_classes)
#     heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
#                                  rotation=0, ha='right')
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
#                                  rotation=45, ha='right')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
# if __name__ == '__main__':
#     net = CNN()
#     # print(net)
#     batch_size = 64
#     train_path = "../dataset/Task2_dataset/train"
#     test_path = "../dataset/Task2_dataset/test"
#
#     num_epochs = 10
#
#     #加载数据
#     #将train数据集分为验证集和训练集
#     train_loader, eval_loader= load_train_data(batch_size,train_path)
#     test_data,test_label = load_test_data(batch_size,test_path)
#     #训练模型
#     train(net, train_loader, eval_loader,device, num_epochs)
#     # #测试模型-画出混淆矩阵
#     plot_confusion_matrix(test_data,test_label,net)
#     if hasattr(torch.cuda, 'empty_cache'):
#         torch.cuda.empty_cache()


# JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
# cnn_acc_all = [0.8654871977240398, 0.871354907539118, 0.8601529160739687, 0.8677098150782361, 0.8682432432432432, 0.8723328591749644, 0.9278982930298719, 0.9100284495021337, 0.9456792318634424, 0.9807076813655761, 0.9998221906116643, 0.9996443812233285, 0.9997332859174964, 0.9999110953058321]
# cldnn_acc_all = [0.9355440967283073, 0.9250533428165008, 0.8758001422475107, 0.9150071123755334, 0.9254089615931721, 0.8990931721194879, 0.9181187766714083, 0.9421230440967283, 0.9671052631578947, 0.9951102418207681, 0.9984886201991465, 0.9989331436699858, 0.9992887624466572, 0.9991998577524893]
# cnncat_acc_all = [0.8685099573257468, 0.8693100995732574, 0.8613975817923186, 0.8699324324324325, 0.8722439544807966, 0.8795341394025604, 0.8964260312944523, 0.9122510668563301, 0.9451458036984353, 0.999377667140825, 0.9998221906116643, 0.9997332859174964, 0.9998221906116643, 0.9985775248933144]
# cnnlstm_acc_all = [0.8628200568990043, 0.8685988620199147, 0.870199146514936, 0.8637091038406828, 0.8673541963015647, 0.8750889046941679, 0.9190967283072546, 0.9311877667140825, 0.9584815078236131, 0.997599573257468, 0.9971550497866287, 0.9997332859174964, 0.999377667140825, 0.9999110953058321]
# plt.figure()
# plt.plot(JSR,cnn_acc_all, "ro-",label="CNN Test Acuracy")
# plt.plot(JSR,cldnn_acc_all, "bs-",label="CLDNN Test Acuracy")
# plt.plot(JSR,cnncat_acc_all, "g.-",label="CNN_cat Test Acuracy")
# plt.plot(JSR,cnnlstm_acc_all, "y*-",label="CNN_LSTM Test Acuracy")
# plt.legend(bbox_to_anchor=(1.00,0.25))
# #加网格线
# plt.grid()
# plt.ylim(0.0, 1.1)
# plt.xlabel("JSR")
# plt.ylabel("Accuracy")
# plt.title("Average Classification Accuracy")
# plt.show()


import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    print(Y.shape[2:])
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

if __name__ == '__main__':
    X = torch.rand(8, 8)
    X = X.view((1,) + X.shape)
    print(X.shape[2:])

