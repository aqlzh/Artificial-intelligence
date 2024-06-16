import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        # 520卷积
        self.cnn1 = nn.Conv2d(in_channels=3,
                              out_channels=16,
                              kernel_size=5,
                              stride=2,
                              padding=0)

        # 311卷积
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # 710卷积
        self.cnn3 = nn.Conv2d(in_channels=32,
                              out_channels=128,
                              kernel_size=7,
                              stride=1,
                              padding=0)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 激活函数
        self.relu = nn.ReLU()
        # 线性输出层
        self.fc = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # [8, 3, 32, 32] -> [8, 16, 14, 14]
        x = self.cnn1(x)
        x = self.relu(x)

        # [8, 16, 14, 14] -> [8, 32, 14, 14]
        x = self.cnn2(x)
        x = self.relu(x)

        # [8, 32, 14, 14] -> [8, 32, 7, 7]
        x = self.pool(x)

        # [8, 32, 7, 7] -> [8, 128, 1, 1]
        x = self.cnn3(x)
        x = self.relu(x)

        # 展平，便于线性计算
        x = x.flatten(start_dim=1)

        # [8,128] -> [8,10]
        return self.fc(x)


def getModel():
    return CnnNet()


if __name__ == '__main__':
    net = getModel()
    print(net(torch.rand(8, 3, 32, 32)).shape)
