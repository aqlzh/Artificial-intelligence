import torch
import numpy as np
import cv2

# print(torch.from_numpy(np.zeros([2, 2])))

img_data = cv2.imread('../data/test1.png')
print(type(img_data),img_data.shape)
cv2.imshow('test1', img_data)
cv2.waitKey(0)
# numpy转tensor
tensor_img_data = torch.from_numpy(img_data)
print(tensor_img_data)

# 把数据放到GPU上
out = tensor_img_data.to(torch.device('cuda:0'))
print('=================================',out.is_cuda)


# tensor转numpy
numpy_img_data = tensor_img_data.numpy()
print(numpy_img_data)