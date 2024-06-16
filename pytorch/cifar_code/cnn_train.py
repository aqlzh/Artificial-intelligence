import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from cnn_net import getModel
from cnn_load_cifar10 import train_data_loader

# 超参数
epoch_num = 300
lr = 1e-3

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

        for i, (x, y) in enumerate(train_data_loader):
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_func(out, y)

            loss.backward()  # 根据loss计算模型的梯度
            optimizer.step()  # 根据梯度调整模型的参数
            optimizer.zero_grad()  # 梯度归0，进行下一轮计算

            acc = (out.argmax(dim=1) == y).sum().item() / len(y)
            print(
                'epoch:{},step:{}/{},loss:{},acc:{}'.format(epoch + 1, i + 1, len(train_data_loader), loss.item(), acc))

        torch.save(net, f'../model/net_epoch{epoch + 1}.pkl')


if __name__ == '__main__':
    train()
