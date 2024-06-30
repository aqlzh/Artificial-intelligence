import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a, b)

# 算术运算
print(torch.add(a, b))  # 加
print(torch.sub(a, b))  # 减
print(torch.mul(a, b))  # 哈代码积
print(torch.div(a, b))  # 除法

# 矩阵运算
a = torch.ones(2, 1)
b = torch.ones(1, 2)
print(torch.matmul(a, b))  # 矩阵乘法

# 高维tensor只能用matmul(最后两个维度要可以进行矩阵运算)
a = torch.ones(1, 2, 3, 4, 5, 6, 7, 8)
b = torch.ones(1, 2, 3, 4, 5, 6, 8, 7)
print(torch.matmul(a, b))  # 矩阵乘法
print(torch.matmul(a, b).shape)  # 矩阵乘法

# 指数运算
a = torch.tensor([1, 2])
print(torch.pow(a, 2))

# exp
print(torch.exp(a))

# tensorboard_log
print(torch.log(a))  # 以e为底
print(torch.log2(a))  # 以2为底
print(torch.log10(a))  # 以10为底

# 开方
print(torch.sqrt(a))
