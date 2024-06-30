import torch

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a, b, sep='\n')
# torch.where
print(torch.where(a > 0.5, a, b))
# torch.index_select
print(torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2])))
# torch.gather
c = torch.linspace(1, 16, 16).view(4, 4)
print(torch.gather(c, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                 [0, 1, 2, 2],
                                                 [0, 1, 3, 3]])))
# torch.mask
mask = torch.gt(c,8)
print(mask)
print(torch.masked_select(c, mask))