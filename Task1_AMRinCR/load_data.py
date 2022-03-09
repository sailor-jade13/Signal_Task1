'''
@Project ：PytorchTest
@File ：load_data.py
@IDE  ：PyCharm
@Author ：Jade
@Date ：2021/11/4 18:47
'''
import torch
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

# data_shape = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape
    # data_shape = (14,300000)
    # 每个信号,含14个信噪比,每个信噪比下有300000个数据

# 信号分类
radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1', 'gfsk1', 'pam4', '16qam', '64qam']
# QAM信号分类
diagram_classes = ['16qam', '64qam']

#signal_dataset
# 训练集：验证集 = 7:3
def load_train_data_from_mat(data_path, snr_index=None, test_split=0.3):
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
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
            train_ind = int(points / 128 * (1 - test_split))
            for i in range(int(points / 128)):
                # 测试集
                if i > train_ind - 1:
                    eval_data.append(dat[:, i * 128:(i + 1) * 128, :])
                    if index == 7:
                        # QAM信号不进行区分，放在一起
                        eval_label.append(index - 1)
                    else:
                        eval_label.append(index)
                # 训练集
                else:
                    train_data.append(dat[:, i * 128:(i + 1) * 128, :])
                    if index == 7:
                        train_label.append(index - 1)
                    else:
                        train_label.append(index)
        # print(np.array(train_data).shape)
    return np.array(train_data), np.array(train_label), np.array(eval_data), np.array(eval_label)


#signal_dataset
#处理测试集
def load_test_data_from_mat(data_path, snr_index=None):
    test_data = []
    test_label = []
    # 若给出信噪比，则对应数据集中相应的信噪比分量
    if snr_index or snr_index < 1:
        snr = [snr_index]
    # 若不给出信噪比，则求平均信噪比
    else:
        snr = range(14)
    points = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape[1]
    # print(points)
    for index, src in enumerate(radio_classes):
        for s in snr:
            dat = np.zeros((2, points, 1))
            for i, typ in enumerate(['im', 're']):
                path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
                dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][s]
            for i in range(int(points / 128)):
                test_data.append(dat[:, i * 128:(i + 1) * 128, :])
                if index == 7:
                    test_label.append(index - 1)
                else:
                    test_label.append(index)
        # print(np.array(test_data).shape)
    return np.array(test_data), np.array(test_label)


#constellation_diagram
def load_data_from_jpg(data_path,snr_index=None):
    if snr_index or snr_index<1:
        snr = [snr_index]
    else:
        snr = range(14)
    data = []
    label = []
    for index,src in enumerate(diagram_classes):
        for s in snr:
            path = os.path.join(data_path,src,'snr_{}'.format(s+1))
            # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            for file in os.listdir(path):
                img = Image.open(os.path.join(path,file))
                data.append(np.array(img)/255.0)
                label.append(index)
        return np.array(data),np.array(label)

#constellation_diagram
#数据预处理
def load_data_from_jpg(data_path, snr_index=None):
    # 若给出信噪比，则对应数据集中相应的信噪比分量
    if snr_index or snr_index < 1:
        snr = [snr_index]
    # 若不给出信噪比，则求平均信噪比的精确度
    else:
        snr = range(14)
    data = []
    label = []
    for index, src in enumerate(diagram_classes):
        for s in snr:
            dir_path = os.path.join(data_path, src, 'snr_{}'.format(s + 1))
            # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            for file in os.listdir(dir_path):
                img = Image.open(os.path.join(dir_path, file))
                data.append(np.array(img)/255.0)
                label.append(index)
    # print (np.array(data).shape, np.array(label).shape)
    return np.array(data), np.array(label)

# if __name__ == '__main__':
#     path = "../dataset/Test1_dataset/constellation_diagram/train"
#     load_data_from_jpg(path,13)
#
