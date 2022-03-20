'''
@Project ：Signal_Task 
@File ：wrong_test.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/17 21:41 
'''
# 不同干扰信号测试准确度
# def signal_acc(test_loader, net, device=None):
#     net.eval()
#     acc_cwi = torch.FloatTensor()
#     acc_scwi = torch.FloatTensor()
#     acc_lfmi = torch.FloatTensor()
#     acc_pi = torch.FloatTensor()
#     acc_nbi = torch.FloatTensor()
#     acc_wbi = torch.FloatTensor()
#     acc_csi = torch.FloatTensor()
#     num = 0
#     # n1,n2,n3,n4,n5,n6,n7 = 0,0,0,0,0,0,0
#     for step, (b_x, b_y) in enumerate(test_loader):
#         with torch.no_grad():
#             test_data = b_x.float().cuda()
#             test_label = b_y.long().cuda()
#             output = net(test_data)
#             pre_lab = torch.argmax(output, 1)
#             if (test_label == 0)& (pre_lab ==0):
#                 acc_cwi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 1)& (pre_lab ==1):
#                 acc_scwi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 2)& (pre_lab ==2):
#                 acc_lfmi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 3)& (pre_lab ==3):
#                 acc_pi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 4)& (pre_lab ==4):
#                 acc_nbi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 5)& (pre_lab ==5):
#                 acc_wbi += test_label.shape[0]
#                 num += test_label.shape[0]
#             if (test_label == 6)& (pre_lab ==6):
#                 acc_csi += test_label.shape[0]
#                 num += test_label.shape[0]
#             # acc_cwi += ((test_label == 0)& (pre_lab ==0) ).float().sum().cpu().item()
#             # acc_scwi += ((test_label == 1) & (pre_lab == 1)).float().sum().cpu().item()
#             # acc_lfmi += ((test_label == 2) & (pre_lab == 2)).float().sum().cpu().item()
#             # acc_pi += ((test_label == 3) & (pre_lab == 3)).float().sum().cpu().item()
#             # acc_nbi += ((test_label == 4)& (pre_lab ==4)).float().sum().cpu().item()
#             # acc_wbi += ((test_label == 5) & (pre_lab == 5)).float().sum().cpu().item()
#             # acc_csi += ((test_label == 6)& (pre_lab ==6)).float().sum().cpu().item()
#     return acc_cwi/num,acc_scwi/num,acc_lfmi/num,acc_pi/num,acc_nbi/num,acc_wbi/num,acc_csi/num

# def signal_acc(test_loader, my_convnet, device=None):
#     net.eval()
#     if device is None and isinstance(my_convnet, torch.nn.Module):
#         device = list(net.parameters())[0].device
#     acc_cwi,acc_scwi,acc_lfmi,acc_pi,acc_nbi,acc_wbi,acc_csi = 0.0,0.0,0.0,0.0,0.0,0.0,0.0
#     n = 0
#     for step, (b_x, b_y) in enumerate(test_loader):
#         # 测试函数前加了装饰器，解决了cuda out of memory
#         with torch.no_grad():
#             if isinstance(my_convnet, torch.nn.Module):
#                 test_data = b_x.float().to(device)
#                 test_label = b_y.long().to(device)
#                 output = my_convnet(test_data)
#                 pre_lab = torch.argmax(output, 1)
#                 acc_cwi += ((test_label == 0)& (pre_lab ==0) ).float().sum().cpu().item()
#                 acc_scwi += ((test_label == 1) & (pre_lab == 1)).float().sum().cpu().item()
#                 acc_lfmi += ((test_label == 2) & (pre_lab == 2)).float().sum().cpu().item()
#                 acc_pi += ((test_label == 3) & (pre_lab == 3)).float().sum().cpu().item()
#                 acc_nbi += ((test_label == 4)& (pre_lab ==4)).float().sum().cpu().item()
#                 acc_wbi += ((test_label == 5) & (pre_lab == 5)).float().sum().cpu().item()
#                 acc_csi += ((test_label == 6)& (pre_lab ==6)).float().sum().cpu().item()
#                 n += test_label.shape[0]
#     return acc_cwi/n,acc_scwi/n,acc_lfmi/n,acc_pi/n,acc_nbi/n,acc_wbi/n,acc_csi/n


from scipy import io
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9])
x = x.reshape(3,3)
print(x)
print(x.shape)

io.savemat('SavedData.mat',{'x':x})
