import visdom
import torch
import cv2
import os

# os.system('python -m visdom.server')

# 使用前先启动服务
# python -m visdom.server

# 解决端口占用问题
# netstat -aon|findstr 8097
# taskkill /f /pid  15932

# print(visdom.__file__)
vis = visdom.Visdom()
# vis.text('hello world')
image = cv2.imread('../data/test1.png')
print(image.shape)
print(image.transpose(2, 0, 1)[::-1, ...].shape)
# 需要把通道数放到前面
vis.image(image.transpose(2, 0, 1)[::-1, ...])


# 注意cv2的图片是BGR，用visdom显示出来是RGB的，而且通道数在第一位。需要用图示方法进行转换。...代表剩下的维度们，也可以放在前面，代表前面的维度们。
