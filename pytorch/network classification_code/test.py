# test train  相结合
import torch
from load_data import dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load('../model/net1.pkl').to(device)


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
