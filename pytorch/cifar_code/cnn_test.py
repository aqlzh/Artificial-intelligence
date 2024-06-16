import torch
from cnn_load_cifar10 import test_data_loader
import glob

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


@torch.no_grad()  # 不计算模型梯度
def test_single(net):
    net.eval()  # 进入测试模式

    acc = 0
    for i, (x, y) in enumerate(test_data_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x).argmax(dim=1)
        acc += (out == y).sum().item()
    return acc


def test():
    modes_name = glob.glob('../model/net_epoch*.pkl')
    for mode_name in modes_name:
        net = torch.load(mode_name).to(device)
        acc = test_single(net)
        print(modes_name, acc/len(test_data_loader.dataset))


if __name__ == '__main__':
    test()
