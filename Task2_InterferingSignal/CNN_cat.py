'''
@Project ：Signal_Task 
@File ：CNN_cat.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/10 9:38 
'''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
# 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

#读取数据并对数据进行预处理
#将train数据集(300000)按比例7:3分为train和eval两部分
def load_train_data(batch_size,path_train,snr):
    train_data, train_label, eval_data, eval_label = load_train_data_from_mat(path_train,snr)
    train_data = torch.unsqueeze(torch.from_numpy(train_data),dim=1)
    eval_data = torch.unsqueeze(torch.from_numpy(eval_data), dim=1)
    train_data = torch.unsqueeze(train_data, dim=2)
    eval_data = torch.unsqueeze(eval_data, dim=2)
    print("train_data.shape:",train_data.shape)  #(26248,1,1,64) - 300000/64*8*0.7
    print("eval_data.shape:", eval_data.shape)   #(2400,1,1,64) - 300000/64*8*0.3

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
    print("test_data.shape:", test_data.shape)  # (11248,1,1,64)

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
    return test_data,test_label

#网络参数初始化
class CNN_cat(nn.Module):
    def __init__(self):
        super(CNN_cat, self).__init__()
        #2层CNN
        self.conv1 = nn.Sequential(
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
        self.conv2 = nn.Sequential(
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
            nn.Conv2d(64, 32, (1, 7), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, (1, 7), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        #1层全连接
        self.fc = nn.Sequential(
            nn.Linear(2*16*1*52,128),
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
    def forward(self, x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x1, x2], dim=2)
        # print(x.shape)
        output = self.fc(x.view(x.shape[0], -1))
        return output

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
            #分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
            b_x = b_x.float().to(device)
            #分类问题中，pytorch默认的预测标签是64位有符号整型数据
            b_y = b_y.long().to(device)
            net.train()
            output = net(b_x,b_x)
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
            # 分类问题中，使用Pytorch需要的数据为torch的32位浮点型张量
            b_x = b_x.float().to(device)
            # 分类问题中，pytorch默认的预测标签是64位有符号整型数据
            b_y = b_y.long().to(device)
            net.eval()
            output = net(b_x,b_x)
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


#测试模型-画出混淆矩阵
def plot_confusion_matrix(test_data,test_label,my_convnet):
    with torch.no_grad():
        test_data = test_data.float().to(device)
        test_label = test_label.long().to(device)
        my_convnet.eval()
        output = my_convnet(test_data,test_data)
        # 返回指定维度最大值的序号下标
        pre_lab = torch.argmax(output, 1)
        #test准确度
        test_acc = accuracy_score(test_label.cpu(), pre_lab.cpu())
        print(f"test_acc:{test_acc}")
        conf_mat = confusion_matrix(test_label.cpu(), pre_lab.cpu())
        df_cm = pd.DataFrame(conf_mat, index=signal_classes, columns=signal_classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                     rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                     rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

#平均测试准确度
def test_accuracy(test_data,test_label,my_convnet):
    with torch.no_grad():
        test_data = test_data.float().to(device)
        test_label = test_label.long().to(device)
        my_convnet.eval()
        output = my_convnet(test_data)
        # 返回指定维度最大值的序号下标
        pre_lab = torch.argmax(output, 1)
        # test准确度
        test_acc = accuracy_score(test_label.cpu(), pre_lab.cpu())
        print(f"test_acc:{test_acc}")
        return test_acc

# 不同干扰信号测试准确度
def evaluate_accuracy(test_data,test_label, my_convnet, device=None):
    if device is None and isinstance(my_convnet, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_cwi,acc_scwi,acc_lfmi,acc_pi,acc_nbi,acc_wbi,acc_csi = 0.0,0.0,0.0,0.0,0.0,0.0,0.0
    n = 0
    # 测试函数前加了装饰器，解决了cuda out of memory
    with torch.no_grad():
        if isinstance(my_convnet, torch.nn.Module):
            test_data = test_data.float().to(device)
            test_label = test_label.long().to(device)
            my_convnet.eval()
            output = my_convnet(test_data)
            acc_cwi += ((test_label == 0)& (my_convnet(test_data).argmax(dim=1) ==0) ).float().sum().cpu().item()
            acc_scwi += ((test_label == 1) & (my_convnet(test_data).argmax(dim=1) == 1)).float().sum().cpu().item()
            acc_lfmi += ((test_label == 2) & (my_convnet(test_data).argmax(dim=1) == 2)).float().sum().cpu().item()
            acc_pi += ((test_label == 3) & (my_convnet(test_data).argmax(dim=1) == 3)).float().sum().cpu().item()
            acc_nbi += ((test_label == 4)& (my_convnet(test_data).argmax(dim=1) ==4)).float().sum().cpu().item()
            acc_wbi += ((test_label == 5) & (my_convnet(test_data).argmax(dim=1) == 5)).float().sum().cpu().item()
            acc_csi += ((test_label == 6)& (my_convnet(test_data).argmax(dim=1) ==6)).float().sum().cpu().item()
        n += test_label.shape[0]
        n = n/8
    return acc_cwi/n,acc_scwi/n,acc_lfmi/n,acc_pi/n,acc_nbi/n,acc_wbi/n,acc_csi/n

if __name__ == '__main__':
    input_dim = 60
    hidden_dim = 128
    layer_dim = 1
    output_dim = 8
    net = CNN_cat()
    print(net)

    batch_size = 64
    train_path = "../dataset/Task2_dataset/train"
    test_path = "../dataset/Task2_dataset/test"
    num_epochs = 40

    #图1-混淆矩阵
    #加载数据
    #将train数据集分为验证集和训练集
    train_loader, eval_loader= load_train_data(batch_size,train_path,4)
    test_data,test_label = load_test_data(batch_size,test_path,4)
    # #训练模型
    my_convnet,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # # #测试模型-画出混淆矩阵
    plot_confusion_matrix(test_data,test_label,my_convnet)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    # test_acc_all = []
    # JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    # for i in range(0,14):
    #     print(f"SNR:{-8+i*2}")
    #     train_loader, eval_loader= load_train_data(batch_size,train_path,i)
    #     test_data,test_label = load_test_data(batch_size,test_path,i)
    #     #训练模型
    #     my_convnet,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    #     #测试准确度
    #     test_acc = test_accuracy(test_data,test_label,my_convnet)
    #     test_acc_all.append(test_acc)
    #     print(test_acc_all)
    #     if hasattr(torch.cuda, 'empty_cache'):
    #         torch.cuda.empty_cache()
    # plt.figure()
    # plt.plot(JSR,test_acc_all, "ro-",label="Test acc")
    # plt.legend(bbox_to_anchor=(1.00,0.1))
    # #加网格线
    # plt.grid()
    # plt.ylim(0.0, 1.1)
    # plt.xlabel("JSR")
    # plt.ylabel("Accuracy")
    # plt.title("Average Classification Accuracy")
    # plt.show()

    # 图3-各自准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    # test_acc_cwi = []
    # test_acc_scwi = []
    # test_acc_lfmi = []
    # test_acc_pi = []
    # test_acc_nbi = []
    # test_acc_wbi = []
    # test_acc_csi = []
    # JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    # for i in range(0,14):
    #     train_loader, eval_loader= load_train_data(batch_size,train_path,i)
    #     test_data,test_label = load_test_data(batch_size,test_path,i)
    #     #训练模型
    #     my_convnet,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    #     #测试准确度
    #     cwi_acc,scwi_acc,lfmi_acc,pi_acc,nbi_acc,wbi_acc,csi_acc = evaluate_accuracy(test_data,test_label,my_convnet)
    #     test_acc_cwi.append(cwi_acc)
    #     test_acc_scwi.append(scwi_acc)
    #     test_acc_lfmi.append(lfmi_acc)
    #     test_acc_pi.append(pi_acc)
    #     test_acc_nbi.append(nbi_acc)
    #     test_acc_wbi.append(wbi_acc)
    #     test_acc_csi.append(csi_acc)
    # plt.figure()
    # plt.plot(JSR,test_acc_cwi, "bo-",label="CWI")
    # plt.plot(JSR, test_acc_scwi, "m.-", label="SCWI")
    # plt.plot(JSR, test_acc_lfmi, "g*-", label="LFMI")
    # plt.plot(JSR,test_acc_pi, "y--",label="PI")
    # plt.plot(JSR, test_acc_nbi, "r:", label="NBI")
    # plt.plot(JSR, test_acc_wbi, "k>", label="WBI")
    # plt.plot(JSR, test_acc_csi, "c^", label="CSI")
    # plt.legend(bbox_to_anchor=(0.9,0.5))
    # #加网格线
    # plt.grid()
    # plt.ylim(0.0, 1.1)
    # plt.xlabel("JSR")
    # plt.ylabel("Accuracy")
    # plt.title("Average Classification Accuracy")
    # plt.show()
