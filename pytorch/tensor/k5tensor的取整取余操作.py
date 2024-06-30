import torch

a = torch.rand(2, 2)
a = torch.mul(a, 10)
print(a)

# 向下取整
print(torch.floor(a))
# 向上取整
print(torch.ceil(a))
# 取整数部分
print(torch.trunc(a))
# 取小数部分
print(torch.frac(a))
# 四舍五入
print(torch.round(a))
# 取余
# print(a % 2)