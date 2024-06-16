# 完成对数据的加载,加载到内存中
import glob
import torch

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import csv

label_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


# print(label_dict)
# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            # print(im_item)
            # ../data/train\airplane
            # \aeroplane_s_000004.png
            # im_label_name = im_item.split('\\')[-2]
            im_label_name = im_item.split('/')[-2]
            # print(im_label_name)
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob('../data/train/*/*.png')
im_test_list = glob.glob('../data/test/*/*.png')

train_dataset = MyDataset(im_train_list, transform=transforms.ToTensor())
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())

train_data_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=6)
test_data_loader = DataLoader(dataset=test_dataset, shuffle=True, num_workers=6)


def getdata():
    x = torch.empty(len(train_data_loader), 3072)
    y = torch.empty(len(train_data_loader))

    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.squeeze().view(1, -1)
        x[i] = inputs
        y[i] = labels

    test_x = torch.empty(len(test_data_loader), 3072)
    test_y = torch.empty(len(test_data_loader))

    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.squeeze().view(1, -1)
        test_x[i] = inputs
        test_y[i] = labels

    return x, y.int(), test_x, test_y.int()


def knn(_x, x, y, k):
    temp = _x - x
    temp = torch.pow(temp, 2)
    temp = temp.sum(dim=1)
    temp = torch.sqrt(temp)
    argsort = temp.argsort()  # [3 2 1 0]
    result = y[argsort][:k]  # [1 1 0]
    result = result.int()
    return torch.bincount(result).argmax()  # 取众数


if __name__ == '__main__':

    log_path = '../csv/k.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow([f'k', 'Accuracy'])

    x, y, test_x, test_y = getdata()
    # print(y)
    # print(test_y)
    # 进行测试
    for k in range(1, 301):
        correct = 0
        for i in range(len(test_x)):
            pred = knn(test_x[i], x, y, k=k)
            if pred == test_y[i]:
                correct += 1
            print(f'k={k},step:{i + 1}/{len(test_x)},pred={pred},y={test_y[i]},correct={correct / (i + 1)}')
        print('k={},correct={}'.format(k, correct / len(test_x)))
        csv_writer.writerow([k, correct / len(test_x)])
    file.close()
