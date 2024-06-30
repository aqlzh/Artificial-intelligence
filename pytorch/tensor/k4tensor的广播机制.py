import torch

a = torch.rand(2,1,1,3)
b = torch.rand(4,2,3)

print(a.shape)#2x3
print(b.shape)#1x3
print(torch.add(a,b).shape) #2x3