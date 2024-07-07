import torch

# torch.set_default_device('cuda:0')

# 创建张量
a = torch.Tensor([[1, 2], [3, 4]])
print(a, type(a), a.type(), a.shape)

b = torch.Tensor(2, 3)
print(b, type(b), b.type(), b.shape)

c = torch.ones(2, 2)
print(c, type(c), c.type(), c.shape)

d = torch.eye(2, 3)
print(d, type(d), d.type(), d.shape)

e = torch.zeros(2, 2)
print(e, type(e), e.type(), e.shape)

# 定义和a相同shape的全0 tensor
f = torch.zeros_like(a)
print(f, type(f), f.type(), f.shape)

# 随机tensor
g = torch.rand(4, 5)
print(g, type(g), g.type(), g.shape)

# 正态分布的tensor
h = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(h)

# 均匀分布
i = torch.Tensor(2, 3).uniform_(-1, 1)
print(i)

# 生成序列
j = torch.arange(0, 10, 1)
print(j)

# 等间隔划分
k = torch.linspace(2, 10, 3)
print(k)

# 将索引打乱
l = torch.randperm(10)
print(l)
