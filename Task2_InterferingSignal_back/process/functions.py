'''
@Project ：Signal_Task 
@File ：functions.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/18 11:17 
'''
import copy
import seaborn as sns
from matplotlib import pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from Task2_InterferingSignal.process.load_data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 信号分类
# 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

# 网络训练&测试过程
# 1.读取训练集、验证集和测试集
# 2.对模型进行训练
# 3.用训练好的模型对测试集进行测试
# 4.输出test_accuracy
# 5.可视化结果：混淆矩阵、曲线图

#读取train数据集(300000) 按比例7:3分为train和eval两部分
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

#读取test数据集
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

# 测试模型1
def test_accuracy1(test_loader,net):
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

# 测试模型2
def test_accuracy2(test_loader,net):
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

# 混淆矩阵
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