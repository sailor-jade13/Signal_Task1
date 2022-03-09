'''
@Project ：Signal_Task 
@File ：cat_test.py
@IDE  ：PyCharm 
@Author ：Jade
@Date ：2022/3/6 12:14 
'''
class Connection_Net(nn.Module):
    def __init__(self):
        super(Connection_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

        )
        self.fc = nn.Sequential(
            nn.Linear(16*17*16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 24),
        )

    def forward(self, x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        print(x1.shape)
        print(x2.shape)
        x = torch.cat([x1, x2], dim=2)
        print(x.shape)

        x=self.fc(x.view(x.shape[0], -1))

        return x




class Connection_Net(nn.Module):
    def __init__(self):
        super(Connection_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(1, 4),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

        )
        self.fc = nn.Sequential(
            nn.Linear(16*1*16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 24),
        )

    def forward(self, x1):
        x1 = self.conv1(x1)
        # x2 = self.conv2(x2)
        # print(x1.shape)
        # print(x2.shape)
        # x = torch.cat([x1, x2], dim=2)
        # print(x.shape)

        x=self.fc(x1.view(x1.shape[0], -1))

        return x