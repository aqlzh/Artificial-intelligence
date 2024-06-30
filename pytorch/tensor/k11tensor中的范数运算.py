import torch

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.rand(1, 1)
print(a, b)

print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))

print(c)
print(torch.norm(c, p=1))
print(torch.norm(c, p=2))
print(torch.norm(c, p=3))
