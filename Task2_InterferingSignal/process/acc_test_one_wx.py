# import CNN
from Task2_InterferingSignal.process.test_data_wx import PRE, Getdata
import torch
import scipy.io as sci
import numpy as np


def test(Model, test_data):

    num_correct = 0
    n = 0


    Model.eval()
    correct_pred = {classname: 0 for classname in radio_classes}
    total_pred = {classname: 0 for classname in radio_classes}

    for step, (b_x,b_y) in enumerate(test_data):
        with torch.no_grad():
            b_x = b_x.float().cuda()
            b_y = b_y.cuda()
            outputs = Model(b_x)
            predictions = outputs.argmax(dim=1)
            for label, prediction in zip(b_y, predictions):
                if label == prediction:
                    correct_pred[radio_classes[label]] += 1
                total_pred[radio_classes[label]] += 1
    i=0
    for classname, correct_count in correct_pred.items():


        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))


        cnn_ACC_one[i, kk] = accuracy
        i = i + 1




if __name__ == '__main__':
    radio_classes = ['8psk', 'bpsk', 'cpfsk1', 'gmsk', 'oqpsk', 'qam16', 'qam64', 'qpsk']
    # radio_classes = ['qpsk']
    # for ii,radio_index in enumerate(radio_classes):
    #     print(radio_index)
    cnn_ACC_one = np.zeros((8, 11))
    for kk in range(11):
        pre = PRE(radio_classes=radio_classes, path='../mod_dataset_22.03.09_class_test', snr_index=kk)
        test_data, test_label = pre.load_srcdata()
        getdata = Getdata(batch_size=512, test_data=test_data, test_label=test_label)
        test_iter = getdata.Loader()
        Rmodel = CNN.CNN().cuda()
        Rmodel.load_state_dict(torch.load("CNN_class_22_03_09{}_{}dB.pth".format('ALLmod', -12 + (kk * 2))))


        test(Model=Rmodel,test_data=test_iter)
        sci.savemat('acc_one.mat', {'acc_one': cnn_ACC_one})



