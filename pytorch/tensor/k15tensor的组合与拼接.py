import torch

torch.set_default_device('cuda:0')

a = torch.zeros(2, 4, dtype=torch.int32)
b = torch.ones(2, 4, dtype=torch.int32)

print(a, b, sep='\n')

# cat拼接
print(torch.cat((a, b), dim=0))
print(torch.cat((a, b), dim=1))

# stack拼接(会增加新维度)
print(torch.stack((a, b), dim=0))
print(torch.stack((a, b), dim=1).size())
print(torch.stack((a, b), dim=1))
print(torch.stack((a, b), dim=1).size())
print(torch.stack((a, b), dim=2))
print(torch.stack((a, b), dim=2).size())