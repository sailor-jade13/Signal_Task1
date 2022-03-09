'''
@Project ：PytorchTest
@File ：wrong_load.py
@IDE  ：PyCharm
@Author ：Jade
@Date ：2021/11/4 18:47
'''
import torch
import os
import numpy as np
import copy
import pandas as pd
from scipy.io import loadmat
from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 信号分类
radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1', 'gfsk1', 'pam4', '16qam', '64qam']
# QAM信号分类
diagram_classes = ['16qam', '64qam']

# data_shape = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape
# data_shape = (14,300000)
# 训练集：每个信号,含14个信噪比,每个信噪比下有300000个数据

# 训练集：测试集 = 7:3
def load_train_val_data_from_mat(data_path, snr_index=None):
    train_data = []
    train_label = []
    #若给出信噪比，则对应数据集中相应的信噪比分量
    if snr_index or snr_index < 1:
        snr = [snr_index]
    #若不给出信噪比，则求平均信噪比
    else:
        snr = range(14)
    # os.path.join()函数：连接两个或更多的路径名组件
    # points为每个信号不同信噪比下的数据个数
    points = loadmat(os.path.join(data_path, 'src_8psk_im.mat'))['src_8psk_im'].shape[1]
    for index, src in enumerate(radio_classes):
        for s in snr:
            dat = np.zeros((2, points, 1))
            for i, typ in enumerate(['im', 're']):
                path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
                dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][s]
            # 训练集的iteration数目
            train_ind = int(points / 128 )
            for i in range(int(train_ind)):
                train_data.append(dat[:, i * 128:(i + 1) * 128, :])
                if index == 7:
                    train_label.append(index - 1)
                else:
                    train_label.append(index)
        # print(np.array(train_data).shape)
    return np.array(train_data), np.array(train_label)


#定义网络的训练过程
def train_model(model,traindataloader,train_rate,criterion,optimizer,num_epochs):
    """
    :param model:网络模型
    :param traindataloader:训练数据集，会切分为训练集和验证集
    :param train_rate:训练集batchsize百分比
    :param criterion:损失函数
    :param optimizer:优化方法
    :param num_epochs:训练的轮数
    """
    # model = model.to(device)
    #计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    #复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(num_epochs):
        print(f"Epoch{epoch}/{num_epochs-1}")
        print('-'*10)
        #每个epoch都分为两个阶段：训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step,(b_x,b_y) in enumerate(traindataloader):
            # b_x = b_x.float().to(device)
            # b_y = b_y.float().to(device)
            if step < train_batch_num:
                #设置模式为训练模式
                model.train()
                output = model(b_x)
                # print(output.shape)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #???
                # print(b_x.size(0))  #64
                train_loss += loss.item()*b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y)
                train_num += b_x.size(0)
            else:
                #设置模式为评估模式
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item()*b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        #计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/train_num)
        print(f"{epoch} Train loss: {train_loss_all[-1]:.4f}  Train Acc:{train_acc_all[-1]:.4f}")
        print(f"{epoch} Val loss: {val_loss_all[-1]:.4f}  Val Acc:{val_acc_all[-1]:.4f}")
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    #使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(data={
            "epoch":range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all":val_loss_all,
            "train_acc_all":train_acc_all,
            "val_acc_all":val_acc_all
        })
    return model,train_process



if __name__ == '__main__':
    train_data, train_label = load_train_val_data_from_mat(
    '../dataset/Test1_dataset/signal_dataset/train', 2)
    # print(train_data.shape,train_label.shape,)   #(18744, 2, 128, 1) (18744,)

    # train1_data, train1_label = load_data_from_jpg(
    #       'dataset/RadioML/constellation_diagram/test', snr_index=13)
    # print(train1_data.shape, train1_label.shape)
    # train_data=np.swapaxes(train_data,3,1)
    # print(train_data.shape)
    # train_data = np.swapaxes(train_data,3,2)
    # print(train_data.shape)
    # print(train_data.shape, train_label.shape)