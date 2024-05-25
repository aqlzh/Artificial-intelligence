# 全连接的神经网络
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=2, out_features=128),  # 全连接层
            nn.ReLU(),  # 激活函数，所有负数归0
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),
            nn.Softmax(dim=1)  # 概率相加得1
        )

    def forward(self, x):
        return self.fc(x)


def getModel():
    return Model()


if __name__ == '__main__':
    model = getModel()
    print(model(torch.randn(8, 2)).shape)
