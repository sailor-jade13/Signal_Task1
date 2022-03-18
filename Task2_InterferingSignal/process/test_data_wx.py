import os
import numpy as np
import h5py
from torch.utils.data import Dataset,DataLoader

class PRE(object):
    def __init__(self,radio_classes,path,snr_index):
        self.radio_classes = radio_classes
        print("调制方式为：{}".format(self.radio_classes))
        self.path = path
        self.snr_index = snr_index
        print("信噪比为{}".format(-12 + (self.snr_index * 2)))

    def load_srcdata(self):

        src_data = []
        radio_srcname = []

        if isinstance(self.radio_classes, list):
            radio_srcname = self.radio_classes
        else:
            radio_srcname.append(self.radio_classes)

        points = h5py.File(os.path.join(self.path, 'src_{}_re.mat'.format(radio_srcname[0])),'r')[
            'src_{}_re'.format(radio_srcname[0])].shape[0]
        print('points大小：', points)
        if self.snr_index or self.snr_index < 1:
            snrr_index = [self.snr_index]
        else:
            snrr_index = [range(11)]

        for index, src in enumerate(radio_srcname):
            for ss in snrr_index:

                dat = np.zeros((1, 2, points))

                for k, typ in enumerate(['im', 're']):

                    path = os.path.join(self.path, 'src_{}_{}.mat'.format(src, typ))
                    datmat = h5py.File(path,'r')['src_{}_{}'.format(src, typ)]
                    dat[0, k, :] = np.transpose(datmat,(1,0))[ss]

                for i in range(int(points / 1024)):
                    src_data.append(dat[:, :, i * 1024:(i + 1) * 1024])

        test_X = np.array(src_data)
        return test_X

    def load_nondata(self):

        non_data = []
        radio_nonname = []

        if isinstance(self.radio_classes, list):
            radio_nonname = self.radio_classes
        else:
            radio_nonname.append(self.radio_classes)

        points = h5py.File(os.path.join(self.path, 'non_{}_re.mat'.format(radio_nonname[0])), 'r')[
            'non_{}_re'.format(radio_nonname[0])].shape[0]

        if self.snr_index or self.snr_index < 1:
            snrr_index = [self.snr_index]
        else:
            snrr_index = [range(11)]

        for index, src in enumerate(radio_nonname):
            for ss in snrr_index:
                dat = np.zeros((1, 2, points))
                for k, typ in enumerate(['im', 're']):

                    path = os.path.join(self.path, 'non_{}_{}.mat'.format(src, typ))
                    datmat = h5py.File(path, 'r')['non_{}_{}'.format(src, typ)]
                    dat[0, k, :] = np.transpose(datmat, (1, 0))[ss]

                for i in range(int(points/1024)):

                    non_data.append(dat[:, :, i * 1024:(i + 1) * 1024])

        test_Y = np.array(non_data)
        return test_Y


class MySet(Dataset):
    def __init__(self, X, Y):

        self.X, self.Y = X, Y

    def __getitem__(self, index):

        return self.X[index], self.Y[index]

    def __len__(self):

        return len(self.X)

class Getdata(object):
    def __init__(self, batch_size, test_X, test_Y):
        self.batch_size = batch_size

        self.test_X = test_X             # X 为有噪声数据
        self.test_Y = test_Y             # Y 为无噪声数据


    def Loader(self):
        TestSet = MySet(self.test_X, self.test_Y)

        test_iter = DataLoader(TestSet,batch_size=self.batch_size,shuffle=False,num_workers=24,
                                pin_memory=True)
        return test_iter


if __name__ == '__main__':
    # radio_classes = ['8psk', 'bpsk', 'cpfsk1', 'gmsk', 'oqpsk', 'qam16', 'qam64', 'qpsk']
    radio_classes = ['gmsk']
    NON_data = np.zeros((2, 512000))
    for ii,radio_index in enumerate(radio_classes):
        print(radio_index)
        for kk in range(11):
            pre = PRE(radio_classes=radio_index,path='../mod_dataset_rls7',snr_index=kk)
            print('第{}次循环'.format(kk+1))
            test_x = pre.load_srcdata()
            test_y = pre.load_nondata()
            getdata = Getdata(batch_size=512,test_X=test_x,test_Y=test_y)
            test_iter = getdata.Loader()
            print(test_x.shape)
            print(test_y.shape)