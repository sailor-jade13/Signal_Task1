'''
@Project ：PytorchTest 
@File ：test1.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2021/11/4 18:52 
'''
import torch
import torch.nn as nn
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
from scipy.io import loadmat
from PIL import Image
from sklearn.metrics import confusion_matrix

# radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1',
#                  'gfsk1', 'pam4', '16qam', '64qam']
# for index, src in enumerate(radio_classes):
#     print(isinstance(src, list))

# snr_index = 13
# if snr_index or snr_index < 1:
#     snrr_index = [snr_index]
# print(snrr_index)


# snrr_index = [13]
# data_path =  '../dataset/Test1_dataset/signal_dataset/train'
# points = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape[1]
# # snrr_index = range(14)
# for index, src in enumerate(radio_classes):
#     for ss in snrr_index:
#         dat = np.zeros((2, points, 1))
#         for i, typ in enumerate(['im', 're']):
#             path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
#             dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)]
#             # print(dat[i, :, 0].shape)


# snrr_index = [13]
# data_path = '../dataset/Test1_dataset/signal_dataset/test'
# # print(loadmat('../dataset/Test1_dataset/signal_dataset/train/src_64qam_im').values())
# points = loadmat(os.path.join(data_path, 'src_64qam_im.mat'))['src_64qam_im'].shape
# print(points)
# shape = (14,300000)
# 每一种信号共30000个数据？？
# 还是每一个数据针对不同的信噪比有30000个数据？？

# # snrr_index = range(14)
# snr_index = 10
# if snr_index or snr_index < 1:
#     snrr_index = [snr_index]
# for index, src in enumerate(radio_classes):
#     for ss in snrr_index:
#         # print(ss) # 10 10 10 10 10 10 10 10
#                   # 对8种信号的该信噪比进行提取
#         #同一信号同一信噪比下信号的2个分量：正交分量和同相分量
#         dat = np.zeros((2, points, 1))
#         for i, typ in enumerate(['im', 're']):
#             path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
#             # print(loadmat(path)['src_{}_{}'.format(src, typ)].shape)
#             dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][ss]
#         print(dat[0,:30,0])


# dataset = loadmat('../dataset/Test1_dataset/signal_dataset/train/src_64qam_im')
# # print(dataset.keys())
# #dict_keys(['__header__', '__version__', '__globals__', 'src_64qam_im'])
# #前3个是公用的，每个mat文件都会有
# data_train = dataset.get('src_64qam_im')
# # print(data_train.shape)  #(14, 300000)
# print(data_train[:,0:10])


# t = torch.tensor([[1,2],[3,4],[2,8]])
# print(torch.argmax(t,0))
# print(torch.argmax(t,1))


# a = np.array([[1.5, 6.7], [6.8, 3.4]])
# b = torch.from_numpy(a)
# f = nn.Softmax(dim=0)
# c = f(b)
# print(c)
# # tensor([[0.0050, 0.9644],
# #        [0.9950, 0.0356]], dtype=torch.float64)
# f = nn.Softmax(dim=1)
# c = f(b)
# print(c)
# # tensor([[0.0055, 0.9945],
# #         [0.9677, 0.0323]], dtype=torch.float64)


a = np.array([[[1.5, 6.7, 2.4],
                [6.8, 3.4, 9.3]],
              [[3.1, 6.5, 1.9],
                [8.9, 1.2, 2.5]]])
b = torch.from_numpy(a)
f = nn.Softmax(dim=0)
c = f(b)
print(c)
#tensor([[[0.1680, 0.5498, 0.6225],
        #  [0.1091, 0.9002, 0.9989]],

        # [[0.8320, 0.4502, 0.3775],
        #  [0.8909, 0.0998, 0.0011]]], dtype=torch.float64)
#0.1680+0.8320 = 1，即dim = 0，是让两个2x3数据的对应位置和为1
f = nn.Softmax(dim=1)
c = f(b)
print(c)
# tensor([[[0.0050, 0.9644, 0.0010],
#          [0.9950, 0.0356, 0.9990]],

#         [[0.0030, 0.9950, 0.3543],
#          [0.9970, 0.0050, 0.6457]]], dtype=torch.float64)
# 0.0050+0.9950 = 1,即dim = 1，是让张量每个2x3数据自己的列之和为1
torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=None, profile=None)
f = nn.Softmax(dim=2)
c = f(b)
print(c)
# tensor([[[5.41325e-03, 9.81272e-01, 1.33145e-02],
#          [7.56666e-02, 2.52524e-03, 9.21808e-01]],

#         [[3.19843e-02, 9.58382e-01, 9.63350e-03],
#          [9.97890e-01, 4.51872e-04, 1.65805e-03]]], dtype=torch.float64)
# dim=2，就是让张量每个2x3数据自己的行之和为1