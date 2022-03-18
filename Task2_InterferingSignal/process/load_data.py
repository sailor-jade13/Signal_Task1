'''
@Project ：Signal_Task 
@File ：load_data.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/1/7 9:38
'''
import torch
import os
import numpy as np
from scipy.io import loadmat

train_data_path = "../../dataset/Task2_dataset/train"
test_data_path = "../../dataset/Task2_dataset/test"

# 信号分类
# 无干扰  单音干扰 多音干扰 线性扫频 脉冲干扰 窄带干扰 宽带干扰 梳状谱干扰
signal_classes = ['bpsk','cwi','scwi','lfmi','pi','nbi','wbi','csi']

data_shape = loadmat(os.path.join(train_data_path, 'src_bpsk_jsr.mat'))['src_bpsk_jsr'].shape
# print(data_shape)
# data_shape = (14, 300000)
# 每个信号,含14个信噪比,每个信噪比下有300000个数据

# 训练集：验证集 = 7:3
def load_train_data_from_mat(train_data_path, snr_index=None, eval_split=0.3):
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    # 若给出信噪比，则对应数据集中相应的信噪比分量
    if snr_index or snr_index < 1:
        snr = [snr_index]
        # 若不给出信噪比，则求平均信噪比
    else:
        snr = range(14)
    # os.path.join()函数：连接两个或更多的路径名组件
    # points为每个信号不同信噪比下的数据个数 —— 300000
    points = loadmat(os.path.join(train_data_path, 'src_bpsk_jsr.mat'))['src_bpsk_jsr'].shape[1]
    for index, src in enumerate(signal_classes):
        for s in snr:
            dat = np.zeros((points))
            path = os.path.join(train_data_path, 'src_{}_jsr.mat'.format(src))
            dat[:] = loadmat(path)['src_{}_jsr'.format(src)][s].real
            # 训练集的iteration数目
            train_ind = int(points / 64 * (1 - eval_split))
            for i in range(int(points / 64)):
                # 测试集
                if i > train_ind - 1:
                    eval_data.append(dat[i * 64:(i + 1) * 64])
                    eval_label.append(index)
                # 训练集
                else:
                    train_data.append(dat[i * 64:(i + 1) * 64])
                    train_label.append(index)
            # print(np.array(train_data).shape)
    return np.array(train_data), np.array(train_label), np.array(eval_data), np.array(eval_label)

#signal_dataset
#处理测试集
def load_test_data_from_mat(test_data_path,snr_index=None):
    test_data = []
    test_label = []
    # 若给出信噪比，则对应数据集中相应的信噪比分量
    if snr_index or snr_index < 1:
        snr = [snr_index]
    # 若不给出信噪比，则求平均信噪比
    else:
        snr = range(14)
    points = loadmat(os.path.join(test_data_path, 'src_bpsk_jsr.mat'))['src_bpsk_jsr'].shape[1]
    # print(points) #90000
    for index, src in enumerate(signal_classes):
        for s in snr:
            dat = np.zeros((points))
            path = os.path.join(test_data_path, 'src_{}_jsr.mat'.format(src))
            dat[:] = loadmat(path)['src_{}_jsr'.format(src)][s].real
            for i in range(int(points / 64)):
                test_data.append(dat[i * 64:(i + 1) * 64])
                test_label.append(index)
            # print(np.array(test_data).shape)
    return np.array(test_data), np.array(test_label)

if __name__ == '__main__':
    train_data,train_label,eval_data,eval_label = load_train_data_from_mat(train_data_path,13)
    test_data,test_label = load_test_data_from_mat(test_data_path,13)
    #300000*8  /64 =18750  - 训练集:验证集=7:3= 26248：11248
    print(train_data.shape) #(26248, 64)
    print(train_label.shape) #(26248,)
    print(eval_data.shape)  #(11248, 64)
    print(eval_label.shape)  #(11248,)
    print(test_data.shape) #(11248, 64)
    print(test_label.shape) #(11248,)