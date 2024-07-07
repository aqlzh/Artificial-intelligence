import torch

torch.set_default_device('cuda:0')

a = torch.floor(torch.rand(3, 4) * 100)

print(a)
# 平均切片
print(torch.chunk(a, 2, dim=0))
print(torch.chunk(a, 2, dim=1))

# 非平均切片
print(torch.split(a, 2, dim=0))  # 以2为单位进行切片，最后一个可以不满2
print(torch.split(a, 2, dim=1))
print(torch.split(a, [1, 2, 1], dim=1))
