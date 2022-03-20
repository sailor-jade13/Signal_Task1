'''
@Project ：Signal_Task 
@File ：VGG.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/18 15:33 
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
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        #4层CNN
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(1,2),
            nn.Conv2d(32, 64, (1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(64, 128, (1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(128, 128, (1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(1, 2),
        )
        #2层全连接+1层输出
        self.hidden2 = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn. ReLU(),
            nn. Dropout(0.9),
            nn.Linear(256, 8),
        )
    def forward(self, x):
        x = self.hidden1(x)
        # print(x.shape)
        output = self.hidden2(x.view(x.shape[0], -1))
        return output

# 不同信号准确度测试
def test(model, test_loader):
    model.eval()
    correct_pred = {classname: 0 for classname in signal_classes}
    total_pred = {classname: 0 for classname in signal_classes}
    for step, (b_x,b_y) in enumerate(test_loader):
        with torch.no_grad():
            b_x = b_x.float().cuda()
            b_y = b_y.cuda()
            output = model(b_x)
            pre_lab = output.argmax(dim=1)
            # 将对象中对应的元素打包成一个个元组
            for label, pre_lab in zip(b_y, pre_lab):
                if label == pre_lab:
                    correct_pred[signal_classes[label]] += 1
                total_pred[signal_classes[label]] += 1
    i=0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))
        cnn_ACC_one[i, kk] = accuracy
        i = i + 1

if __name__ == '__main__':
    net = VGG()
    # print(net)
    batch_size = 64
    train_path = "../../dataset/Task2_dataset/train"
    test_path = "../../dataset/Task2_dataset/test"

    num_epochs = 30

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
    # test_accuracy1(test_loader, net)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # 图2-平均准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    test_acc_all = []
    JSR = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    for i in range(14):
        print(f"JSR:{-8 + i * 2}")
        train_loader, eval_loader= load_train_data(batch_size,train_path,i)
        test_loader = load_test_data(batch_size,test_path,i)
        #训练模型
        net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
        #测试准确度
        test_acc = test_accuracy1(test_loader,net)
        test_acc_all.append(test_acc)
        print(test_acc_all)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    plt.figure()
    plt.plot(JSR,test_acc_all, "ro-",label="process acc")
    # plt.legend(bbox_to_anchor=(1.00,0.1))
    plt.legend()
    #加网格线
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.xlabel("JSR")
    plt.ylabel("Accuracy")
    plt.title("Average Classification Accuracy")
    plt.show()

    # 图3-不同类型干扰的识别准确度曲线
    # # 横坐标—干信比 纵坐标-平均识别准确度
    # cnn_ACC_one = np.zeros((8, 14))
    # JSR = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    # for kk in range(14):
    #     print(f"JSR:{-8 + kk * 2}")
    #     train_loader, eval_loader = load_train_data(batch_size, train_path, kk)
    #     test_loader = load_test_data(batch_size, test_path, kk)
    #     # 训练模型
    #     net, train_process = train(net, train_loader, eval_loader, device, num_epochs)
    #     # 测试准确度
    #     test(net, test_loader)
    #     # io.savemat('acc_one.mat', {'acc_one': cnn_ACC_one})
    # print(cnn_ACC_one)
    # plt.figure()
    # plt.plot(JSR, cnn_ACC_one[1, :] / 100, "bo-", label="CWI")
    # plt.plot(JSR, cnn_ACC_one[2, :] / 100, "m.-", label="SCWI")
    # plt.plot(JSR, cnn_ACC_one[3, :] / 100, "g*-", label="LFMI")
    # plt.plot(JSR, cnn_ACC_one[4, :] / 100, "yD-", label="PI")
    # plt.plot(JSR, cnn_ACC_one[5, :] / 100, marker='*', linestyle='-', label="NBI")
    # plt.plot(JSR, cnn_ACC_one[6, :] / 100, "k.-", label="WBI")
    # plt.plot(JSR, cnn_ACC_one[7, :] / 100, "r^-", label="CSI")
    # # plt.legend(bbox_to_anchor=(0.9,0.5))
    # xticks(np.linspace(-8, 18, 14, endpoint=True))
    # plt.legend()
    # # #加网格线
    # plt.grid()
    # # plt.ylim(0.0, 1.1)
    # plt.xlabel("JSR")
    # plt.ylabel("Accuracy")
    # plt.title("Average Classification Accuracy")
    # plt.show()
