'''
@Project ：Signal_Task 
@File ：ResNet18.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/15 10:45 
'''
# ResNet-18
# 每个模块里有4个卷积层（不计算1×1卷积层），加上最开始的卷积层和最后的全连接层，共计18层

import torch.nn.functional as F
import copy
import seaborn as sns
from matplotlib import pyplot as plt
from pylab import xticks
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from Task2_InterferingSignal.process.load_data import *
from Task2_InterferingSignal.process.functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
#                无干扰 单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

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

# 不同干扰信号测试准确度
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
    net = resnet18(8)
    print(net)

    batch_size = 64
    train_path = "../../dataset/Task2_dataset/train"
    test_path = "../../dataset/Task2_dataset/test"
    num_epochs = 20

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
    plt.legend(bbox_to_anchor=(1.00,0.1))
    #加网格线
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.xlabel("JSR")
    plt.ylabel("Accuracy")
    plt.title("Average Classification Accuracy")
    plt.show()

    # 图3-不同类型干扰的识别准确度曲线
    # 横坐标—干信比 纵坐标-平均识别准确度
    # cnn_ACC_one = np.zeros((8, 14))
    # JSR = [-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    # for kk in range(14):
    #     print(f"JSR:{-8 + kk * 2}")
    #     train_loader, eval_loader= load_train_data(batch_size,train_path,kk)
    #     test_loader = load_test_data(batch_size,test_path,kk)
    #     #训练模型
    #     net,train_process = train(net, train_loader, eval_loader,device, num_epochs)
    #     #测试准确度
    #     test(net, test_loader)
    #     # io.savemat('acc_one.mat', {'acc_one': cnn_ACC_one})
    # print(cnn_ACC_one)
    # plt.figure()
    # plt.plot(JSR,cnn_ACC_one[1,:]/100, "bo-",label="CWI")
    # plt.plot(JSR, cnn_ACC_one[2,:]/100, "m.-", label="SCWI")
    # plt.plot(JSR, cnn_ACC_one[3,:]/100, "g*-", label="LFMI")
    # plt.plot(JSR,cnn_ACC_one[4,:]/100, "yD-",label="PI")
    # plt.plot(JSR, cnn_ACC_one[5,:]/100, marker='*',linestyle='-', label="NBI")
    # plt.plot(JSR, cnn_ACC_one[6,:]/100, "k.-", label="WBI")
    # plt.plot(JSR, cnn_ACC_one[7,:]/100, "r^-", label="CSI")
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
