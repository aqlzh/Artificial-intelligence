import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from model import getModel
from load_data import dataloader

# 超参数
epoch_num = 300
lr = 10e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net
net = getModel().to(device)
# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# loss
loss_func = nn.CrossEntropyLoss()


def train():
    net.train()
    for epoch in range(epoch_num):

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_func(out, y)

            loss.backward()  # 根据loss计算模型的梯度
            optimizer.step()  # 根据梯度调整模型的参数
            optimizer.zero_grad()  # 梯度归0，进行下一轮计算

        if epoch % 20 == 0:
            acc = (out.argmax(dim=1) == y).sum().item() / len(y)
            print(epoch, loss.item(), acc)

    # torch.save(net, '../model/net1.pkl')


if __name__ == '__main__':
    train()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = torch.load('../model/net1.pkl').to(device)


@torch.no_grad()  # 不计算模型梯度
def test():
    net.eval()  # 进入测试模式

    acc = 0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        out = net(x).argmax(dim=1)

        acc += (out == y).sum().item()
    print(acc / len(dataloader.dataset))


if __name__ == '__main__':
    test()