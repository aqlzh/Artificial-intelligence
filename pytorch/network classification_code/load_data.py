import torch
from torch.utils.data import DataLoader, Dataset

from generate_data import get_rectangle


class MyDataset(Dataset):

    # 执行数据的加载处理等工作
    def __init__(self):
        pass

    # 定义数据的条数
    def __len__(self):
        return 5000

    # 重载[]
    def __getitem__(self, i):
        a, b, fat = get_rectangle()

        x = torch.FloatTensor([a, b])
        y = fat

        return x, y


dataset = MyDataset()
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True,
                        num_workers=4, drop_last=True)  # drop_last=True不足batch_size直接丢弃

if __name__ == '__main__':
    print(len(dataloader))
    print(next(iter(dataloader)))