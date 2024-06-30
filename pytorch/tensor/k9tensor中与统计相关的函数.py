import torch

a = torch.rand(3, 4)
# 求平均值
print(torch.mean(a))
# 求和
print(torch.sum(a))
# 求积
print(torch.prod(a))
# 求max
print(torch.max(a))
# 求min
print(torch.min(a))
# 求最大值的索引
print(torch.argmax(a))
# 求最小值的索引
print(torch.argmin(a))

# 求标准差
print(torch.std(a))
# 求方差
print(torch.var(a))
# 求中位数
print(torch.median(a))
# 求众数
print(torch.mode(a))

# 计算input的直方图
print(torch.histc(a))
# 返回每个值的频数
b = torch.randint(0, 10, [10])  # 只能处理一维度的tensor
print(b)
print(torch.bincount(b))
