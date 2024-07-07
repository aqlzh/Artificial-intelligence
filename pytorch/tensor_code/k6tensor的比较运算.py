import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a, b)
print(torch.eq(a, b))  # 只比较元素
print(torch.equal(a, b))  # 形状和元素都相同
print(torch.ge(a, b))  # >=
print(torch.gt(a, b))  # >
print(torch.le(a, b))  # <=
print(torch.lt(a, b))  # <
print(torch.ne(a, b))  # !=

# 排序
c = torch.randperm(12)
print(c)
# 从小到大排序
print(torch.sort(c))
# 从大到小排序
print(torch.sort(c, descending=True))
d = c.reshape(3, 4)
print(d)
# 指定维度排序，按列升序
print(torch.sort(d, dim=0))
# 前k个最大元素
print(torch.topk(c, k=2, dim=0))
# 第k小的元素
print(torch.kthvalue(c, k=2, dim=0))

# 判断有界finite无界inf与nan
f = torch.rand(2, 3)
print(torch.isfinite(f))
print(torch.isfinite(f / 0))
print(torch.isinf(f / 0))
print(torch.isnan(f))
f[0, 0] = torch.nan
print(f)
print(torch.isnan(f))
