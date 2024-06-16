# 完成对数据的加载,加载到内存中
import glob
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image

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
            im_label_name = im_item.split('\\')[-2]
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

train_data_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4,drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=4)


if __name__ == '__main__':

    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        print(labels)
        break

