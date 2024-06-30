import torch

# dev = torch.device('cpu')
dev = torch.device('cuda:0')
a = torch.tensor([2, 2], dtype=torch.float32, device=dev)
print(a)

# 定义稀疏张量（非0元素坐标，和具体的值）
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
b = torch.sparse_coo_tensor(i, v, (4, 4),dtype=torch.float32,device=dev)
print(b)
print(b.to_dense())  # 转成稠密的张量
